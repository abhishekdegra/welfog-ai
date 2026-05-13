# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework.permissions import IsAuthenticated


# from .ai import ask_llama, decide_action
# # from .logic import search_products, get_user_orders


# class ChatView(APIView):
#     permission_classes = [IsAuthenticated]

#     def post(self, request):
#         message = request.data.get("message")
#         user = request.user

#         action_data = decide_action(message)

#         action = action_data.get("action")
#         query = action_data.get("query", message)

#         # PRODUCT SEARCH
#         if action == "search_products":
#             result = search_products(query)

#         # ORDER FETCH
#         elif action == "get_orders":
#             result = get_user_orders(user)

#         # GENERAL AI
#         else:
#             return Response({
#                 "reply": ask_ai(message)
#             })

#         # CLEAN RESPONSE FORMAT
#         if result["type"] == "text":
#             return Response({
#                 "reply": result["data"]
#             })

#         elif result["type"] == "product_list":
#             products = result["data"]

#             text = "Here are some products:\n"
#             for p in products:
#                 text += f"- {p['name']} (₹{p['price']})\n"

#             return Response({
#                 "reply": text.strip()
#             })

#         elif result["type"] == "order_list":
#             orders = result["data"]

#             text = "Your orders:\n"
#             for o in orders:
#                 text += f"- {o['product']} ({o['status']})\n"

#             return Response({
#                 "reply": text.strip()
#             })

#         # fallback (safety)
#         return Response({
#             "reply": "Something went wrong."
#         })