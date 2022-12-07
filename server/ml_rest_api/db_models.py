import sys
import os

# для настройки баз данных
from sqlalchemy import Column, ForeignKey, Integer, String, DateTime, JSON, LargeBinary
from sqlalchemy.sql.expression import text
from sqlalchemy.orm import sessionmaker

# для определения таблицы и модели
from sqlalchemy.ext.declarative import declarative_base

# для создания отношений между таблицами
from sqlalchemy.orm import relationship

# для настроек
from sqlalchemy import create_engine

# создание экземпляра declarative_base
Base = declarative_base()

# здесь добавим классы

# создает экземпляр create_engine в конце файла
engine = create_engine(
    f"postgresql+psycopg2://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@database:5432/{os.environ['POSTGRES_DB']}"
)

Base.metadata.create_all(engine)


class MlModel(Base):
    __tablename__ = "saved_models"
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_class = Column(String(50), nullable=False)
    hyperparameters = Column(JSON, nullable=True)
    model = Column(LargeBinary, nullable=False)
    date_added = Column(DateTime, server_default=text("NOW()"))


Base.metadata.bind = engine
Base.metadata.create_all(engine)

DBSession = sessionmaker(bind=engine)
session = DBSession()

modelTest = MlModel(model_class="Linreg", hyperparameters={"1": 1}, model=b"c")
session.add(modelTest)
session.commit()
