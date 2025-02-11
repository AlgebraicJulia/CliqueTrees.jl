""" Some description of ths package
"""
module AlgebraicTemplate

export hello

using Catlab

""" hello(name::String)

Returns the string "Hello, <name>!" where `<name>` is replaced with the provided parameter
"""
hello(name::String) = string("Hello, ", name, "!")

end
