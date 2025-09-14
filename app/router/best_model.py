from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.encoders import jsonable_encoder
from motor.motor_asyncio import AsyncIOMotorCollection

from app.database import get_best_model_collection
from app.schemas import InputBestModel, OutputBestModel
from app.utils.machine_learning import find_best_model
from app.utils.panic import Panic
from app.router.utils import normalize_doc

router = APIRouter(prefix="/best-model")


@router.get("/info")
def best_model_info():
    """
    Thông tin về API tìm mô hình tốt nhất
    """
    return {
        "message": "Đây là nơi bạn có thể tìm mô hình tốt nhất cho dữ liệu của bạn mà không cần dự đoán"
    }


@router.post("/", response_model=OutputBestModel)
async def best_model_post(
    input_data: InputBestModel,
    collection: AsyncIOMotorCollection = Depends(get_best_model_collection),
):
    """
    Tìm mô hình tốt nhất cho dữ liệu đầu vào

    Param:
    - input_data: Dữ liệu đầu vào với X_array, Y_array và x0 (tùy chọn)
    - collection: Bộ sưu tập MongoDB để lưu kết quả

    Return:
    - Kết quả với mô hình tốt nhất và các chỉ số liên quan
    """
    result = find_best_model(
        X=input_data.X_array, Y=input_data.Y_array, x0=input_data.x0
    )

    result["time"] = datetime.now()

    await collection.insert_one(result)

    total = await collection.count_documents({})

    if total > 100:
        excess = total - 100
        old_records_cursor = collection.find().sort("time", 1).limit(excess)
        old_records = await old_records_cursor.to_list(length=excess)
        ids_to_delete = [doc["_id"] for doc in old_records]
        await collection.delete_many({"_id": {"$in": ids_to_delete}})

    return OutputBestModel(**result)


@router.get("/history")
async def best_model_history(
    limit: int = Query(20, gt=1, lt=100),
    skip: int = Query(0, ge=0),
    collection: AsyncIOMotorCollection = Depends(get_best_model_collection),
):
    """
    Lấy lịch sử các lần tìm mô hình tốt nhất

    Param:
    - limit: Số lượng bản ghi trả về (mặc định 20, tối đa 100)
    - skip: Số lượng bản ghi bỏ qua (mặc định 0)
    - collection: Bộ sưu tập MongoDB để truy vấn

    Return:
    - Danh sách các bản ghi lịch sử
    """

    cursor = collection.find().sort("time", -1).skip(skip).limit(limit)
    raw_docs = await cursor.to_list(length=limit)
    normalized_docs = [normalize_doc(doc) for doc in raw_docs]
    return jsonable_encoder(normalized_docs)
