import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app
from app.database import connect_to_database, disconnect_to_database

payload = {
    "X_array": [[50, 1], [60, 2], [70, 2], [80, 3], [90, 3], [100, 4]],
    "Y_array": [150_000, 180_000, 200_000, 220_000, 250_000, 280_000],
    "x0": [[85, 3]],
    "model": "random_forest",
}


@pytest.mark.asyncio
async def test_option():
    await connect_to_database()  # đảm bảo MongoDB được kết nối

    app = create_app()
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post("/regression/option/", json=payload)

        print("🔍 Response JSON:", response.json())  # debug nếu cần

        assert response.status_code == 200
        assert "rmse_train" in response.json()
        assert "rmse_test" in response.json()
        assert "r2_train" in response.json()
        assert "r2_test" in response.json()

    await disconnect_to_database()
