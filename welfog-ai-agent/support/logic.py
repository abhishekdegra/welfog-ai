# from .models import Product, Order

# def search_products(query):
#     products = Product.objects.filter(name__icontains=query)[:5]

#     if not products:
#         return {"type": "text", "data": "No products found."}

#     return {
#         "type": "product_list",
#         "data": [
#             {"name": p.name, "price": p.price}
#             for p in products
#         ]
#     }


# def get_user_orders(user):
#     orders = Order.objects.filter(user=user)

#     if not orders:
#         return {"type": "text", "data": "No orders found."}

#     return {
#         "type": "order_list",
#         "data": [
#             {"product": o.product.name, "status": o.status}
#             for o in orders
#         ]
#     }
# from .models import Product, Order

# def search_products(query):
#     products = Product.objects.filter(name__icontains=query)[:5]

#     if not products:
#         return {"type": "text", "data": "No products found."}

#     return {
#         "type": "product_list",
#         "data": [
#             {"name": p.name, "price": p.price}
#             for p in products
#         ]
#     }


# def get_user_orders(user):
#     orders = Order.objects.filter(user=user)

#     if not orders:
#         return {"type": "text", "data": "No orders found."}

#     return {
#         "type": "order_list",
#         "data": [
#             {
#                 "product": o.product.name,
#                 "status": o.status
#             } for o in orders
#         ]
#     }