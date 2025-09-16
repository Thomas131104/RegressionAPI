from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import RedirectResponse
from app.utils.panic import Panic

router = APIRouter()


@router.get("/history/{type}")
async def get_history(
    type: str,
    limit: int = Query(10, gt=1, lt=100),
    skip: int = Query(0, ge=0),
):
    match type.lower():
        case "option":
            return RedirectResponse(
                url=f"/regression/option/history?limit={limit}&skip={skip}",
                status_code=status.HTTP_302_FOUND,
            )
        case "best_model":
            return RedirectResponse(
                url=f"/regression/best-model/history?limit={limit}&skip={skip}",
                status_code=status.HTTP_302_FOUND,
            )
        case "stacking_model":
            return RedirectResponse(
                url=f"/regression/stack-model/history?limit={limit}&skip={skip}",
                status_code=status.HTTP_302_FOUND,
            )
        case _:
            raise HTTPException(status_code=404, detail="Không tồn tại đường dẫn đó")

    return Panic.unreachable()
