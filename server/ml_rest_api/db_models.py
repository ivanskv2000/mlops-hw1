import os
from sqlalchemy import Column, Integer, String, DateTime, JSON, LargeBinary
from sqlalchemy.sql.expression import text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine

Base = declarative_base()

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

    def todict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}


Base.metadata.bind = engine
Base.metadata.create_all(engine)

DBSession = sessionmaker(bind=engine)
session = DBSession()
