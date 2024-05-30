class User:
    def __init__(
        self,
        first_name: str,
        last_name: str,
        email: str,
        phone: int,
        gender: str,
        address: str,
        role: str,
        password: str = None,
        matric_no: int = None,
        img_path: str = None,
    ) -> None:
        self.first_name = first_name
        self.last_name = last_name
        self.gender = gender
        self.role = role
        self.phone = phone
        self.address = address
        self.email = email
        self.password = password
        self.matric_no = matric_no
        self.img_path = img_path

    def has_permission(self, permission_code, user):
        pass
