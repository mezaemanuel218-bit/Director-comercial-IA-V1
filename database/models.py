from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Lead(Base):
    __tablename__ = "leads"

    id = Column(String, primary_key=True)
    name = Column(String)
    company = Column(String)
    email = Column(String)
    phone = Column(String)
    owner = Column(String)


class Contact(Base):
    __tablename__ = "contacts"

    id = Column(String, primary_key=True)
    name = Column(String)
    email = Column(String)
    phone = Column(String)
    owner = Column(String)


class Account(Base):
    __tablename__ = "accounts"

    id = Column(String, primary_key=True)
    company_name = Column(String)
    industry = Column(String)
    website = Column(String)
    owner = Column(String)


class Note(Base):
    __tablename__ = "notes"

    id = Column(String, primary_key=True)
    parent_id = Column(String)
    parent_type = Column(String)
    content = Column(Text)
    owner = Column(String)
    created_time = Column(DateTime)