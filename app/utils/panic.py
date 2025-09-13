class Panic(Exception):
    """Tương tự Rust panic! — báo lỗi nghiêm trọng."""

    @staticmethod
    def unreachable(message: str = "Code này không thể tới"):
        """
        Tương đương macro `unreachable!()` trong Rust
        """
        raise Panic(f"Unreachable: {message}")

    @staticmethod
    def unimplemented(message: str = "Chưa triển khai"):
        """
        Tương đương macro `unimplenent!()` trong Rust
        """
        raise Panic(f"Unimplemented: {message}")

    @staticmethod
    def todo(message: str = "Cần hoàn tất"):
        """
        Tương đương macro `todo!()` trong Rust
        """
        raise Panic(f"Todo: {message}")
