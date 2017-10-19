class A:
    def __init__(self):
        print("Enter A")
        print("Leave A")

class B(A):
    def __init__(self):
        print("Enter B")
        A.__init__(self)
        print("Leave B")

class C(A):
    def __init__(self):
        print("Enter C")
        A.__init__(self)
        print("Leave C")

class D(A):
    def __init__(self):
        print("Enter D")
        A.__init__(self)
        print("Leave D")

class E(B, C, D):
    def __init__(self):
        print("Enter E")
        B.__init__(self)
        C.__init__(self)
        D.__init__(self)
        print("Leave E")

E()