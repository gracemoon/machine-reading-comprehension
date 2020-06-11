
# def test():
#     global p
#     p=1
#     print(p)


# def testA():
#     global p
#     p=p+1
#     print(p)


# test()
# testA()


import torch 

a=torch.arange(0,6)
print(a)


b=a.unsqueeze(0)
print(b)


c=a.unsqueeze(1)
print(c)


d=a.unsqueeze(-1)
print(d)

e=a.unsqueeze(-2)
print(e)