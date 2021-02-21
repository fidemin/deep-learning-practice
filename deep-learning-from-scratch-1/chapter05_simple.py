

from core.layers import MulLayer

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()


apple_price = mul_apple_layer.forward(100, 2)
final_price = mul_tax_layer.forward(apple_price, 1.1)
print(final_price)

dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)
