from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text,
    JSON, ForeignKey, UniqueConstraint, Index,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, relationship
import uuid


class Base(DeclarativeBase):
    pass


class Property(Base):
    __tablename__ = "properties"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    address = Column(String(500), nullable=False)
    postcode = Column(String(10), index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    uprn = Column(String(20), unique=True, nullable=True)
    property_type = Column(String(50))
    tenure = Column(String(20))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sales = relationship("PropertySale", back_populates="property", cascade="all, delete-orphan")
    reports = relationship("PropertyReport", back_populates="property", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_properties_postcode", "postcode"),
        Index("ix_properties_lat_lng", "latitude", "longitude"),
    )


class PropertySale(Base):
    __tablename__ = "property_sales"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    property_id = Column(UUID(as_uuid=True), ForeignKey("properties.id"), nullable=False)
    transaction_id = Column(String(50), unique=True)
    price = Column(Integer, nullable=False)          # in pence
    date_of_transfer = Column(DateTime)
    property_type = Column(String(10))
    old_new = Column(String(1))                      # N=new build, E=established
    duration = Column(String(1))                     # F=freehold, L=leasehold
    postcode = Column(String(10))
    created_at = Column(DateTime, default=datetime.utcnow)

    property = relationship("Property", back_populates="sales")


class PropertyReport(Base):
    __tablename__ = "property_reports"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    property_id = Column(UUID(as_uuid=True), ForeignKey("properties.id"), nullable=False)
    address = Column(String(500))

    # Raw data snapshots
    land_registry_data = Column(JSON)
    epc_data = Column(JSON)
    crime_data = Column(JSON)
    demographics_data = Column(JSON)
    flood_risk_data = Column(JSON)
    planning_data = Column(JSON)
    schools_data = Column(JSON)
    transport_data = Column(JSON)

    # AI-generated report features
    investment_score = Column(Integer)               # 0–100
    investment_score_reasoning = Column(Text)
    strategy_detector = Column(JSON)                 # {recommended: [...], reasoning: ""}
    renovation_predictor = Column(JSON)
    floorplan_analysis = Column(JSON)
    neighbourhood_intelligence = Column(JSON)
    rental_demand_score = Column(Integer)
    planning_scanner = Column(JSON)
    deal_finder = Column(JSON)
    price_growth_predictor = Column(JSON)
    rental_yield_simulator = Column(JSON)
    ai_summary = Column(Text)

    generated_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)

    property = relationship("Property", back_populates="reports")


class Portfolio(Base):
    __tablename__ = "portfolios"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(100), nullable=False, index=True)
    name = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)

    holdings = relationship("PortfolioHolding", back_populates="portfolio", cascade="all, delete-orphan")


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    portfolio_id = Column(UUID(as_uuid=True), ForeignKey("portfolios.id"), nullable=False)
    property_id = Column(UUID(as_uuid=True), ForeignKey("properties.id"), nullable=False)
    purchase_price = Column(Integer)                 # in pence
    purchase_date = Column(DateTime)
    strategy = Column(String(50))                    # BTL, HMO, flip, SA, etc.
    monthly_rent = Column(Integer)                   # in pence
    mortgage_payment = Column(Integer)               # in pence
    notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)

    portfolio = relationship("Portfolio", back_populates="holdings")

    __table_args__ = (
        UniqueConstraint("portfolio_id", "property_id", name="uq_holding"),
    )
