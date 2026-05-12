import uvicorn
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from pd_models.order_model import OrderModel
from resources.order import Order, Orders

app = FastAPI()
orders = Orders()
placeRecord = Order()


@app.get('/orders/{id}', operation_id="get_order",
         description="Get an order record. An order has a unique id.")
def get_order(id:str):
    return placeRecord.get(id)


@app.put('/orders/{id}', operation_id="update_order",
            description="Update the rating of an order")
def update_order(id: str, rating: int = 3):
    return placeRecord.update(id, rating)


@app.delete('/orders/{id}', operation_id="delete_order",
            description="Delete or cancel an order. An order has a unique id.")
def delete_order(id: str):
    return placeRecord.delete(id)


@app.post('/orders', operation_id="create_order",
            description="Add an order. An order has a unique id.")
def create_order(order: OrderModel):
    print(order)
    return orders.create(order)


# --- MCP Integration ---
mcp = FastApiMCP(
    app,
    name="Order Management MCP Server",
    description="Tools for managing order records.",
    describe_all_responses=True,
    describe_full_response_schema=True,
    # Only expose the endpoints with these operation_ids
    include_operations=[
        "get_order",
        "create_order",
        "update_order",
        "delete_order",
    ]
)
mcp.mount_http(app, mount_path="/mcp")
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5002, reload=True)
