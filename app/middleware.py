from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response


from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response

class EnforceJSONMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Chỉ kiểm tra Content-Type nếu method có thể có body
        if request.method in ("POST", "PUT", "PATCH"):
            content_type = request.headers.get("Content-Type", "")
            if "application/json" not in content_type:
                return Response(
                    content="❌ Content-Type must be application/json",
                    status_code=415,
                    media_type="application/json"
                )
        return await call_next(request)
