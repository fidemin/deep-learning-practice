
from core.layer import MulLayer, AddLayer

apple = 100
num_apple = 2

orange = 150
num_orange = 3

tax_rate = 1.1

apple_mul_layer = MulLayer()
orange_mul_layer = MulLayer()
apple_orange_add_layer = AddLayer()
tax_layer = MulLayer()

apple_price = apple_mul_layer.forward(apple, num_apple)
orange_price = orange_mul_layer.forward(orange, num_orange)
apple_orange_price = apple_orange_add_layer.forward(apple_price, orange_price)
total_price = tax_layer.forward(apple_orange_price, tax_rate)

print(total_price)


# back propagation
dout = 1

dapple_orange_price, dtax_rate = tax_layer.backward(dout)
dapple_price, dorange_price = apple_orange_add_layer.backward(dapple_orange_price)
dapple, dnum_apple = apple_mul_layer.backward(dapple_price)
dorange, dnum_orange = orange_mul_layer.backward(dorange_price)

print(dapple, dorange)
