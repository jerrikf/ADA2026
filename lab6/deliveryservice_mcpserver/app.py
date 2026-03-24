import uvicorn
from fastapi import FastAPI
from fastapi_mcp import FastApiMCP

from db import Base, engine
from pdmodels.delivery_req import DeliveryReq
from pdmodels.status_req import StatusModel
from resources.delivery import Delivery
from resources.status import Status

app = FastAPI()
Base.metadata.create_all(engine)


@app.post("/deliveries",  operation_id="create_delivery", description="Record a delivery for a specific order by a customer.")
def create_delivery(d_req: DeliveryReq):
    return Delivery.create(d_req)


@app.get('/deliveries/{d_id}', operation_id="get_delivery", description="Get a delivery of a customer. The delivery is identified by its unique ID.")
def get_delivery(d_id: int):
    return Delivery.get(d_id)


@app.put('/deliveries/{d_id}/status', operation_id="update_delivery_status",
         description="Update the staus a delivery. The delivery is identified by its unique ID.")
def update_delivery_status(d_id: int, new_status: StatusModel):
    return Status.update(d_id, new_status.status)


@app.delete('/deliveries/{d_id}', operation_id="delete_delivery",
            description="Delete or remove a delivery. The delivery is identified by its unique ID.")
def delete_delivery(d_id: int):
    return Delivery.delete(d_id)

# --- MCP Integration ---
mcp = FastApiMCP(
    app,
    name="Deliver Record API Services",
    description="Tools for managing order deliveries.",
    describe_all_responses=True,
    describe_full_response_schema=True,
    # Only expose the endpoints with these operation_ids
    include_operations=[
        "create_delivery",
        "get_delivery",
        "update_delivery_status",
        "delete_delivery",
    ]
)
mcp.mount_http(app, mount_path="/mcp")
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)

