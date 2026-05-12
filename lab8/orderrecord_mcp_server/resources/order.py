import random
from fastapi import status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pd_models.order_model import OrderModel

orderRecords = [
    {
        "id": "id1",
        "product_type": "Laptop",
        "quantity": 4000,
        "unit_price": 444.50
    }
]


class Order:
    def get(self, id):
        for record in orderRecords:
            if id == record["id"]:
                return record, 200
        return JSONResponse(content=jsonable_encoder({'message': "There is no order with id " + id}),
                            status_code=status.HTTP_404_NOT_FOUND)

    def update(self, id, rating):
        for record in orderRecords:
            if id == record["id"]:
                record["rating"] = rating
                return record, 200
        return JSONResponse(content=jsonable_encoder({'message': "There is no order with id " + id}),
                            status_code=status.HTTP_404_NOT_FOUND)

    def delete(self, id):
        to_be_deleted = None
        for record in orderRecords:
            if id == record["id"]:
                to_be_deleted = record
                break
        if to_be_deleted:
            orderRecords.remove(to_be_deleted)
            return JSONResponse(content=jsonable_encoder({"message": "{} is deleted.".format(id)}),
                            status_code=status.HTTP_200_OK)
        return JSONResponse(content=jsonable_encoder({'message': "There is no order with id " + id}),
                            status_code=status.HTTP_404_NOT_FOUND)


class Orders:
    def create(self, order_model:OrderModel):
        record_to_be_created = order_model.model_dump(mode="json")
        id1 = record_to_be_created["id"]
        for record in orderRecords:
            if id1 == record["id"]:
                return JSONResponse(content=jsonable_encoder({"message": "Order with id {} already exists".format(id)}),
                                    status_code=status.HTTP_400_BAD_REQUEST)
        orderRecords.append(record_to_be_created)
        return JSONResponse(content=jsonable_encoder(record_to_be_created),
                            status_code=status.HTTP_201_CREATED)

    def get(self):
        results = []
        for record in orderRecords:
            results.append(record["id"])
        return JSONResponse(content=jsonable_encoder(results),
                            status_code=status.HTTP_200_OK)
