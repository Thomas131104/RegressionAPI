import asyncio

import hypercorn.asyncio
from hypercorn.config import Config

from app import create_app

config = Config()
config.bind = ["127.0.0.1:8000"]
config.reload = True

app = create_app()

if __name__ == "__main__":
    asyncio.run(hypercorn.asyncio.serve(app, config))
