from model import R3Net
model = R3Net()
input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, inputs=(input, ))
print("flops:", flops/1000000000)
print("params", float(params/1000000))
