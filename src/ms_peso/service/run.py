from __future__ import annotations

import uvicorn

from ms_peso.service.app import create_app
from ms_peso.service.config import ServiceSettings


def main() -> None:
    settings = ServiceSettings.from_env()
    uvicorn.run(
        create_app(settings=settings),
        host=settings.host,
        port=settings.port,
        workers=1,
    )


if __name__ == "__main__":
    main()
