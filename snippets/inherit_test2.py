class A:
    def __init__(self):
        print("Enter A")
        print("Leave A")

class B(A):
    def __init__(self):
        print("Enter B")
        super(B, self).__init__()
        print("Leave B")

class C(A):
    def __init__(self):
        print("Enter C")
        super(C, self).__init__()
        print("Leave C")

class D(A):
    def __init__(self):
        print("Enter D")
        super(D, self).__init__()
        print("Leave D")

class E(B, C, D):
    def __init__(self):
        print("Enter E")
        super(E, self).__init__()
        print("Leave E")

E()