from pydantic import BaseModel, BaseConfig


class DefaultConfig(BaseConfig):
    allow_population_by_field_name = True
    arbitrary_types_allowed = True


class DefaultBase(BaseModel):
    class Config(DefaultConfig):
        pass
