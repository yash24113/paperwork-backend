from app.core.config import Settings, get_settings

SettingsDep = Settings


def settings_dependency() -> Settings:
    return get_settings()
