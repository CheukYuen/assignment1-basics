"""
Explore chr(0) — the NULL character (U+0000).

(a) What Unicode character does chr(0) return?
(b) How does __repr__() differ from print()?
(c) What happens when it occurs in text?
"""

print("=== (a) What is chr(0)? ===")
c = chr(0)
print(f"chr(0) = {chr(0)}")
print(f"Unicode name: NULL (U+0000)")
print()

print("=== (b) __repr__() vs print() ===")
print(f"repr:  {repr(c)}")      # shows escape sequence
print(f"print: ", end="")
print(c)                     # prints... nothing visible
print()

print(f"repr:  {ord('牛')}") 
print(f"repr:  {chr(29275)}") 
