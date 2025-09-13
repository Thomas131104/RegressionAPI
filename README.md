Input tổng quát:
- X : Mảng 1 chiều | Mảng 2 chiều | Danh sách các dict (Chỉ lấy value)
- Y : Mảng 1 chiều
- x0 : Giá trị x cần dự đoán

Output tổng quát:
- optimized_y0 : Giá trị y từ mô hình học máy có chỉ số RMSE nhỏ nhất
- optimized_model : Mô hình học máy tối ưu với Input
- optimized_index : Chỉ số tối ưu (RMSE)
- optimized_intercept
- optimized_bias

- list_of_ys : Các giá trị y từ các mô hình học máy khác nhau:
    - model : Mô hình học máy
    - y0 : Giá trị y từ mô hình học máy tương ứng
    - index : Chỉ số RMSE với mô hình đó