"""
Explore chr(0) — the NULL character (U+0000).

(a) What Unicode character does chr(0) return?
(b) How does __repr__() differ from print()?
(c) What happens when it occurs in text?
"""

print("=== (a) What is chr(0)? ===")
c = chr(0)
print(f"chr(0) = {c!r}")
print(f"Unicode name: NULL (U+0000)")
print()

print("=== (b) __repr__() vs print() ===")
print(f"repr:  {c!r}")      # shows escape sequence
print(f"print: ", end="")
print(c)                     # prints... nothing visible
print()

print("=== (c) chr(0) in text ===")
s1 = "this is a test" + chr(0) + "string"
print(f"repr:  {s1!r}")
print(f"print: {s1}")
print(f"len('test' + chr(0) + 'string') = {len(s1)}  (the null byte IS there, just invisible)")
print()

# Bonus: encode to bytes to see it clearly
print("=== Bonus: bytes view ===")
print(f"bytes: {s1.encode('utf-8')}")
