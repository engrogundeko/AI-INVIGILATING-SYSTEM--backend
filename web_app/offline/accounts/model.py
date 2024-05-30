# from __future__ import annotations
# from dataclasses import dataclass
# from enum import Enum
# from typing import List, BinaryIO, Dict, Any

# from bson import ObjectId
# from ..repository import userRespository, MongoDBRepository


# @dataclass
# class Service:
#     def create(self, *args, **kwargs): ...
#     def get(self, *args, **kwargs): ...
#     def update(self, *args, **kwargs): ...
#     def delete(self, *args, **kwargs): ...


# class Gender:
#     male = "MALE"
#     female = "FEMALE"


# @dataclass
# class Permission:
#     code: str
#     name: str
#     db: MongoDBRepository = MongoDBRepository(collection_name="permission")


class Role:
    admin = "ADMIN"
    student = "STUDENT"
    lecturer = "LECTURER"
    invigilator = "INVIGILATOR"
#     db: MongoDBRepository = MongoDBRepository(collection_name="role")


# @dataclass
# class UserService:
#     id: ObjectId = None  # Let MongoDB generate this
#     first_name: str = None
#     last_name: str = None
#     email: str = None
#     phone: int = None
#     gender: str = None
#     address: str = None
#     role: Role = None
#     password: str = None
#     matric_no: int = None
#     img_path: str = None
#     permission: Permission = None
#     db: MongoDBRepository = MongoDBRepository(collection_name="user")

#     def data(self) -> Dict[str, Any]:
#         return {
#             "first_name": self.first_name,
#             "last_name": self.last_name,
#             "email": self.email,
#             "phone": self.phone,
#             "gender": self.gender,
#             "address": self.address,
#             "role": self.role,
#             "password": self.password,
#             "matric_no": self.matric_no,
#             "img_path": self.img_path,
#             "permissions": self.permission,
#         }

#     def add(self) -> UserService:
#         data = self.data()
#         inserted_id = self.db.insert_one(data).inserted_id
#         data["id"] = inserted_id
#         return UserService(**data)

#     @property
#     def get(self, *args, **kwargs): ...

#     @property
#     def remove_user(self):
#         return self.db.delete_one({"_id": self.id})

#     def update_user(self, id: ObjectId): ...

#     def get_permissions(self): ...

#     def has_permission(self, permission_code):
#         pass

#     @property
#     def full_name(self):
#         return self.first_name + self.last_name


# payload = None
# user = UserService(payload.__dict__).add()
# user["email"]
# user.email
