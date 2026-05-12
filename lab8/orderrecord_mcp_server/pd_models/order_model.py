from pydantic import BaseModel, Field


class OrderModel(BaseModel):
    id: str = Field(description="Order identifier")
    product_type: str = Field(description="Type of the ordered product")
    quantity: int = Field(description="Ordered quantity of a product")
    unit_price: float = Field(description="The unit price of the ordered product")
