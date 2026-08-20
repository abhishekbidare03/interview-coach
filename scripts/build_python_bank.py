r"""Curated Python question bank — written by hand, not generated.

Why this exists instead of `generate_bank.py --topic python`:

The generated Python bank was unusable for its actual purpose. A 3B model asked
for "an interview question about closures" writes a *written exam* question —
compound, hedged, and forty words long:

    "Can you explain the concept of closures in Python and how they relate to
     late binding? How do closures ensure that inner functions can access outer
     function variables even after the outer function has finished executing?"

No interviewer says that. They say "What's a closure?" and then follow up. Two
questions welded together with an "and" also breaks grading, because the
coverage score can no longer distinguish "answered half of it" from "answered
one of the two things completely".

The second failure was the difficulty spread: 22 questions at difficulty
{2: 8, 3: 5, 4: 9} and *nothing* at 1. Real screening rounds are mostly basics —
you check the floor before you probe the ceiling. This bank is deliberately
bottom-heavy.

Design rules, applied to every entry below:

* **One question, one sentence, under ~18 words.** If it needs an "and", it is
  two questions; the second one becomes a follow-up seed.
* **Expected points are short and independently checkable.** The grader asks
  "did they state this?" per point, so a point that is really three facts in a
  trench coat grades as a coin flip.
* **Reference answers are spoken prose.** They are read aloud when the candidate
  is wrong, so no code blocks, no symbols, and dunder names written the way a
  person says them ("dunder init", not "__init__").

Run:  .venv\Scripts\python.exe scripts\build_python_bank.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bank_kit as kit                                # noqa: E402
from bank_kit import Entry                            # noqa: E402
from coach.schema import Mode                         # noqa: E402

BANK_DATA: list[Entry] = [

    # ---------------------------------------------------------------- 1 ---- #
    # Warm-up. Everyone should clear these; they exist to settle nerves and to
    # find the floor fast when someone cannot.

    (1, "list versus tuple",
     "What's the difference between a list and a tuple?",
     "A list is mutable, so you can add, remove, or change items after you "
     "create it. A tuple is immutable — once it's built, it's fixed. That "
     "immutability is why a tuple can be used as a dictionary key and a list "
     "cannot.",
     ["a list is mutable",
      "a tuple is immutable",
      "tuples can be dictionary keys, lists cannot",
      "tuples suit fixed records, lists suit changing collections"],
     ["Why can a tuple be a dictionary key when a list can't?"]),

    (1, "dictionaries",
     "How is a dictionary different from a list?",
     "A list is ordered and you get items out by their integer position. A "
     "dictionary stores key-value pairs and you get items out by key. Dictionary "
     "lookup by key is roughly constant time, because it's backed by a hash "
     "table.",
     ["a list is indexed by position",
      "a dictionary maps keys to values",
      "dictionary lookup by key is about constant time",
      "a dictionary is backed by a hash table"],
     ["What kinds of objects can you use as a dictionary key?"]),

    (1, "sets",
     "What's a set used for in Python?",
     "A set is an unordered collection of unique items. It automatically drops "
     "duplicates, and membership testing is about constant time rather than "
     "scanning the whole thing. It's the right choice for deduplicating and for "
     "fast 'have I seen this' checks.",
     ["a set holds unique items only",
      "a set is unordered",
      "membership testing is about constant time",
      "useful for deduplicating a collection"],
     ["How would you find the items two lists have in common?"]),

    (1, "identity versus equality",
     "What's the difference between the is operator and double equals?",
     "Double equals compares values — whether two objects mean the same thing. "
     "The is operator compares identity — whether they are literally the same "
     "object in memory. Two separate lists with the same contents are equal but "
     "not identical.",
     ["double equals compares values",
      "is compares object identity",
      "two equal objects can be different objects",
      "use is for None, not double equals"],
     ["Which one should you use to check for None, and why?"]),

    (1, "mutability of strings",
     "Are strings mutable in Python, and what does that mean in practice?",
     "Strings are immutable. Any operation that looks like it changes a string "
     "actually builds a new one. In practice that means building a long string "
     "by repeatedly adding to it in a loop is wasteful — you'd join a list of "
     "pieces instead.",
     ["strings are immutable",
      "string operations return a new string",
      "repeated concatenation in a loop is inefficient",
      "join a list of pieces instead"],
     ["So how would you build a long string efficiently?"]),

    (1, "dynamic typing",
     "What does it mean to say Python is dynamically typed?",
     "Types belong to values, not to variable names, and they're checked when "
     "the code runs rather than ahead of time. The same name can hold an integer "
     "now and a string later. The trade-off is flexibility against errors that "
     "only surface at runtime.",
     ["types are checked at runtime",
      "variables are not declared with a type",
      "a name can be rebound to a different type",
      "type errors surface at runtime rather than earlier"],
     ["Where do type hints fit in, if types aren't enforced?"]),

    (1, "append versus extend",
     "What's the difference between append and extend on a list?",
     "Append adds its argument as a single element, so appending a list gives "
     "you a nested list. Extend iterates its argument and adds each item "
     "individually, so the list grows by that many elements.",
     ["append adds one element",
      "extend adds each item of an iterable",
      "appending a list produces a nested list",
      "extend grows the list by the iterable's length"],
     ["What happens if you extend a list with a string?"]),

    (1, "None",
     "What does None represent in Python?",
     "None is Python's null — a single object meaning 'no value here'. A "
     "function with no return statement returns it. You check for it with the is "
     "operator, because there's only ever one None object.",
     ["None means the absence of a value",
      "a function without a return statement returns None",
      "check for it using is rather than double equals",
      "there is exactly one None object"],
     []),

    (1, "list comprehensions",
     "What's a list comprehension?",
     "It's a compact way of building a list from an iterable in a single "
     "expression, with an optional filter. It's usually more readable than the "
     "equivalent loop with an append, and slightly faster because the looping "
     "happens in C.",
     ["builds a list from an iterable in one expression",
      "can include a filtering condition",
      "replaces a loop that appends",
      "generally faster and more readable than the loop"],
     ["When would a plain loop be the better choice?"]),

    (1, "indentation and blocks",
     "How does Python know where a block of code starts and ends?",
     "Indentation. Where other languages use braces, Python uses the "
     "indentation level itself to define the block, so it's part of the syntax "
     "rather than a style choice. Mixing tabs and spaces inconsistently is a "
     "syntax error.",
     ["indentation defines block structure",
      "there are no braces around blocks",
      "indentation is syntax, not just style",
      "inconsistent indentation raises an error"],
     []),

    (1, "error handling basics",
     "How do you handle an error that might be raised by some code?",
     "You wrap it in a try block and catch what you expect with an except "
     "clause. You should catch the specific exception rather than everything, so "
     "you don't accidentally swallow bugs. A finally block runs either way, for "
     "cleanup.",
     ["wrap the risky code in a try block",
      "catch specific exceptions in an except clause",
      "avoid catching every exception blindly",
      "finally runs whether or not an exception occurred"],
     ["Why is catching every exception a bad habit?"]),

    (1, "range",
     "What does the range function give you in Python 3?",
     "A lazy range object, not a list. It generates values as you iterate, so it "
     "uses the same small amount of memory whether you asked for ten values or "
     "ten million. You'd wrap it in list only if you actually need the list.",
     ["range returns a lazy range object, not a list",
      "values are produced on demand",
      "memory use does not grow with the range size",
      "call list on it if you need an actual list"],
     []),

    (1, "membership testing",
     "What does the in keyword do?",
     "It tests membership. For a list or tuple it scans element by element, so "
     "it's linear. For a set or a dictionary it's a hash lookup, so it's about "
     "constant time. For a string it checks for a substring.",
     ["tests whether an item is in a collection",
      "linear scan for a list or tuple",
      "about constant time for a set or dictionary",
      "checks for a substring when used on a string"],
     ["If you're doing that check in a loop, which structure would you pick?"]),

    (1, "functions and return",
     "What's the difference between printing a value and returning it?",
     "Printing writes to standard output and produces nothing the caller can "
     "use. Returning hands the value back so the caller can store it, pass it "
     "on, or test it. A function that only prints can't be composed or tested "
     "properly.",
     ["print writes to output",
      "return hands a value back to the caller",
      "a printed value cannot be used by the caller",
      "returning makes a function testable and composable"],
     []),

    (1, "built-in types",
     "What are the main built-in data types you use in Python day to day?",
     "Integers, floats, booleans and strings for single values, and lists, "
     "tuples, dictionaries and sets for collections. None sits on its own as the "
     "absence of a value.",
     ["numeric types such as int and float",
      "strings and booleans",
      "collections: list, tuple, dict and set",
      "None as the absence of a value"],
     []),

    (1, "docstrings",
     "What's a docstring and where does it go?",
     "It's a string literal as the very first statement in a module, function, "
     "or class. Python keeps it on the object as the dunder doc attribute, so "
     "help and documentation tools can read it. That's what makes it different "
     "from a comment.",
     ["a string as the first statement in a function, class or module",
      "stored on the object and readable at runtime",
      "used by help and documentation tools",
      "a comment is discarded, a docstring is not"],
     []),

    (1, "slicing",
     "How does slicing work on a list?",
     "You give a start, a stop, and optionally a step. The start is included and "
     "the stop is excluded. Slicing a list gives you a new list — a shallow copy "
     "of that section, not a view onto the original.",
     ["takes a start, stop and optional step",
      "the stop index is excluded",
      "produces a new list",
      "the copy is shallow"],
     ["What does a step of minus one do?"]),

    (1, "loops",
     "When would you use a while loop instead of a for loop?",
     "A for loop is for iterating over something with a known set of items. A "
     "while loop is for repeating until a condition changes, when you don't know "
     "up front how many iterations that takes — reading until end of input, or "
     "retrying until something succeeds.",
     ["for iterates over a known collection or iterable",
      "while repeats until a condition becomes false",
      "use while when the iteration count is not known in advance",
      "while risks looping forever if the condition never changes"],
     []),

    (1, "modules and imports",
     "What's the difference between a module and a package?",
     "A module is a single Python file. A package is a directory of modules that "
     "you can import as a unit. Both are just namespaces — importing one runs it "
     "once and caches it.",
     ["a module is one Python file",
      "a package is a directory containing modules",
      "both act as namespaces",
      "a module is executed once and then cached"],
     []),

    (1, "dict iteration",
     "If you loop over a dictionary directly, what do you get?",
     "The keys. If you want the values you loop over dot values, and if you want "
     "both together you loop over dot items, which gives you key-value pairs. "
     "Since Python 3.7 the order is the insertion order.",
     ["iterating a dict yields its keys",
      "use values to iterate the values",
      "use items to get key-value pairs",
      "insertion order is preserved in modern Python"],
     []),

    # ---------------------------------------------------------------- 2 ---- #
    # Still fundamentals, but the ones that separate "has used Python" from
    # "has read about Python".

    (2, "shallow versus deep copy",
     "What's the difference between a shallow copy and a deep copy?",
     "A shallow copy makes a new outer object but the inner objects are still "
     "shared, so mutating a nested list shows up in both. A deep copy "
     "recursively copies everything, so the two are fully independent. Deep "
     "copying costs more and can trip over cycles.",
     ["a shallow copy shares the nested objects",
      "a deep copy recursively copies nested objects",
      "mutating a nested object affects both shallow copies",
      "deep copy is more expensive"],
     ["Which one does list slicing give you?"]),

    (2, "args and kwargs",
     "What do star args and double star kwargs do in a function signature?",
     "Star args collects any extra positional arguments into a tuple. Double "
     "star kwargs collects any extra keyword arguments into a dictionary. "
     "Together they let a function accept an arbitrary call signature, which is "
     "how wrappers and decorators forward arguments they don't know about.",
     ["star args collects extra positional arguments as a tuple",
      "double star kwargs collects extra keyword arguments as a dict",
      "they allow a variable number of arguments",
      "used to forward arguments through a wrapper"],
     []),

    (2, "decorators",
     "What's a decorator?",
     "A callable that takes a function and returns a replacement, usually one "
     "that wraps the original and adds behaviour around it. The at-sign syntax "
     "is just shorthand for reassigning the name to the wrapped version. Logging, "
     "timing, caching and access checks are the usual uses.",
     ["a function that takes a function and returns a function",
      "wraps the original to add behaviour around it",
      "the at-sign syntax rebinds the name to the wrapper",
      "used for cross-cutting concerns like logging or caching"],
     ["What does functools wraps do, and why would you bother?"]),

    (2, "generators",
     "What's a generator and how does it differ from returning a list?",
     "A generator produces values one at a time as you ask for them, instead of "
     "building the whole sequence up front. Memory stays flat no matter how many "
     "values there are, and you can start consuming before the sequence is "
     "finished. The trade-off is you can only walk it once.",
     ["produces values lazily, one at a time",
      "does not hold the whole sequence in memory",
      "created with yield or a generator expression",
      "can only be iterated once"],
     ["What actually happens the first time you call a generator function?"]),

    (2, "method types",
     "What's the difference between an instance method, a class method and a "
     "static method?",
     "An instance method takes self and works on one object. A class method "
     "takes cls and works on the class itself — that's how you write alternative "
     "constructors. A static method takes neither and is really just a plain "
     "function that lives in the class namespace.",
     ["an instance method receives self",
      "a class method receives the class as cls",
      "a static method receives neither",
      "class methods are useful as alternative constructors"],
     []),

    (2, "default mutable arguments",
     "Why is using a mutable default argument considered a bug waiting to "
     "happen?",
     "Default arguments are evaluated once, when the function is defined, not on "
     "each call. So a default empty list is one single list shared by every call, "
     "and anything appended to it persists into the next call. The fix is to "
     "default to None and create the list inside the function.",
     ["defaults are evaluated once at definition time",
      "the same object is shared across every call",
      "mutations persist between calls",
      "default to None and build the object inside the function"],
     []),

    (2, "context managers",
     "What does the with statement actually do for you?",
     "It guarantees setup and cleanup around a block. On entry it calls the "
     "object's dunder enter, and on exit it calls dunder exit — including when "
     "the block raises. That's why opening a file with it closes the file even "
     "if something blows up halfway through.",
     ["calls dunder enter on entry and dunder exit on exit",
      "cleanup runs even if the block raises",
      "removes the need for a manual try/finally",
      "commonly used for files, locks and connections"],
     ["How would you write a context manager of your own?"]),

    (2, "iterators versus iterables",
     "What's the difference between an iterable and an iterator?",
     "An iterable is anything you can get an iterator from — it implements "
     "dunder iter. An iterator is the thing that actually produces values, one "
     "per call to dunder next, and it's consumed as you go. A list is iterable "
     "but is not itself an iterator.",
     ["an iterable implements dunder iter",
      "an iterator implements dunder next",
      "an iterator is consumed as it is walked",
      "a list is iterable but not an iterator"],
     []),

    (2, "scope",
     "How does Python resolve a variable name inside a function?",
     "It looks in the local scope first, then any enclosing function scopes, "
     "then the module globals, then the builtins — the LEGB order. Assigning to "
     "a name anywhere in a function makes it local for the whole function, which "
     "is what causes surprise unbound errors.",
     ["local, enclosing, global, then builtin order",
      "assignment makes a name local to the whole function",
      "reading an outer name works without any declaration",
      "global or nonlocal is needed to rebind an outer name"],
     ["What does the nonlocal keyword do?"]),

    (2, "sort versus sorted",
     "What's the difference between calling sort and calling sorted?",
     "Sort is a list method that reorders in place and returns None. Sorted is a "
     "built-in that works on any iterable and returns a new list, leaving the "
     "original alone. Assigning the result of sort is a classic way to end up "
     "with None.",
     ["sort sorts a list in place",
      "sort returns None",
      "sorted returns a new list",
      "sorted accepts any iterable"],
     ["How would you sort a list of dictionaries by one field?"]),

    (2, "lambda",
     "What's a lambda, and when is it the right tool?",
     "It's a single-expression anonymous function. It's the right tool when you "
     "need a tiny throwaway callable inline — a sort key, or a default factory. "
     "If it needs a statement, a name, or a docstring, it should be a proper "
     "function.",
     ["an anonymous single-expression function",
      "cannot contain statements",
      "useful inline, for example as a sort key",
      "prefer a named function when the logic is non-trivial"],
     []),

    (2, "f-strings",
     "Why are f-strings usually preferred over the older formatting styles?",
     "They put the expression right where the value appears, so you read the "
     "output in place instead of matching placeholders to arguments further "
     "along. They're also the fastest of the formatting options, because they're "
     "compiled rather than interpreted at runtime.",
     ["expressions are embedded directly in the string",
      "more readable than positional placeholders",
      "faster than percent formatting or format",
      "evaluated at runtime where they appear"],
     []),

    (2, "removing from a list",
     "What's the difference between remove, pop, and del on a list?",
     "Remove deletes the first item equal to the value you give it. Pop deletes "
     "by index and hands the item back. Del deletes by index or slice and returns "
     "nothing. Remove and pop both raise if there's nothing to remove.",
     ["remove deletes by value",
      "pop deletes by index and returns the item",
      "del deletes by index or slice and returns nothing",
      "remove and pop raise when the target is missing"],
     ["What goes wrong if you remove items while looping over the list?"]),

    (2, "dictionary keys",
     "What makes an object usable as a dictionary key?",
     "It has to be hashable — its hash must not change over its lifetime and it "
     "must compare equal consistently. In practice that means immutable types: "
     "strings, numbers, tuples of immutables. Lists and dictionaries can't be "
     "keys because they're mutable.",
     ["the key must be hashable",
      "its hash must not change over its lifetime",
      "immutable types are hashable",
      "lists and dicts cannot be keys"],
     []),

    (2, "truthiness",
     "What counts as false in Python when you put an object in an if statement?",
     "Zero, None, and every empty container — empty string, list, tuple, dict, "
     "set. Everything else is true by default, unless the class defines dunder "
     "bool or dunder len to say otherwise. That's why checking a list for "
     "emptiness doesn't need a length comparison.",
     ["zero and None are false",
      "empty containers are false",
      "everything else is true by default",
      "a class can override this with dunder bool or dunder len"],
     []),

    (2, "exceptions as control flow",
     "Why does Python encourage try/except where other languages check first?",
     "Because attribute and key lookups are cheap and exceptions are cheap when "
     "they don't fire, so asking forgiveness is usually faster and always "
     "race-free. Checking first leaves a window where the thing you checked for "
     "can change before you use it.",
     ["easier to ask forgiveness than permission",
      "avoids a race between the check and the use",
      "exceptions are cheap when not raised",
      "checking first can be slower and more verbose"],
     []),

    (2, "enumerate and zip",
     "How would you loop over a list when you need the index as well as the "
     "item?",
     "Use enumerate, which yields index and item pairs and takes an optional "
     "start. Looping over a range of the length and indexing back in works but is "
     "noisier and easier to get wrong. Zip is the equivalent when you need to "
     "walk two sequences together.",
     ["use enumerate to get index and item together",
      "enumerate accepts a start value",
      "avoids indexing back into the list",
      "zip walks two sequences in parallel"],
     []),

    # ---------------------------------------------------------------- 3 ---- #
    # Where a screening round usually settles.

    (3, "the GIL",
     "What is the GIL, and how does it affect your code?",
     "The global interpreter lock lets only one thread execute Python bytecode "
     "at a time in a single process. So threads don't give you real parallelism "
     "for CPU-bound work — but they're still useful for input and output, "
     "because the lock is released while a thread waits.",
     ["only one thread runs Python bytecode at a time",
      "threads do not give parallel speedup for CPU-bound work",
      "the lock is released during input and output",
      "multiprocessing sidesteps it with separate processes"],
     ["So when would you still reach for threads?"]),

    (3, "threading versus multiprocessing",
     "When would you choose multiprocessing over threading?",
     "For CPU-bound work, because separate processes each get their own "
     "interpreter and their own lock, so they genuinely run in parallel. The cost "
     "is that memory isn't shared, so arguments and results have to be pickled "
     "and copied across.",
     ["multiprocessing suits CPU-bound work",
      "each process has its own interpreter and lock",
      "processes achieve true parallelism",
      "data must be serialised between processes"],
     []),

    (3, "yield",
     "What does yield actually do to a function?",
     "It turns it into a generator function. Calling it runs no body at all — it "
     "returns a generator object. Each time you ask for the next value the body "
     "runs to the next yield and then freezes, keeping its local state until you "
     "come back.",
     ["the function becomes a generator function",
      "calling it does not execute the body",
      "execution pauses at each yield",
      "local state is preserved between resumptions"],
     []),

    (3, "dunder methods",
     "What are dunder methods, and can you name one you've actually used?",
     "They're the hooks the language calls on your behalf — dunder init when an "
     "object is constructed, dunder len for the len function, dunder eq for "
     "equality. Implementing them is how your own class plugs into Python's "
     "built-in syntax instead of needing special-case methods.",
     ["special methods the interpreter calls implicitly",
      "they hook a class into built-in syntax and functions",
      "named a specific example such as dunder init or dunder len",
      "let user classes behave like built-in types"],
     ["If you implement dunder eq, what else should you implement?"]),

    (3, "str versus repr",
     "What's the difference between dunder str and dunder repr?",
     "Dunder repr is for developers — it should be unambiguous and ideally look "
     "like the code that would rebuild the object. Dunder str is for end users "
     "and can be friendlier. If you only write one, write repr, because str "
     "falls back to it.",
     ["repr is aimed at developers and debugging",
      "str is aimed at end-user display",
      "str falls back to repr when not defined",
      "repr should ideally be unambiguous"],
     []),

    (3, "closures",
     "What's a closure?",
     "A function that keeps a reference to variables from the scope it was "
     "defined in, and can still use them after that scope has returned. Python "
     "keeps those variables alive in a cell rather than copying their values, "
     "which is why the closure sees later changes to them.",
     ["an inner function capturing names from an enclosing scope",
      "those names stay usable after the outer call returns",
      "the variable is captured by reference, not by value",
      "the enclosing scope's variables are kept alive"],
     ["What's the classic bug when you build closures in a loop?"]),

    (3, "writing a context manager",
     "How would you write a context manager of your own?",
     "Either implement dunder enter and dunder exit on a class, where enter "
     "returns whatever the as clause binds, or use the contextmanager decorator "
     "on a generator that yields once with the setup before and the cleanup in a "
     "finally.",
     ["implement dunder enter and dunder exit on a class",
      "dunder enter returns the value bound by the as clause",
      "or decorate a generator with contextmanager",
      "cleanup belongs in dunder exit or a finally block"],
     ["How do you make it swallow an exception rather than propagate it?"]),

    (3, "duck typing",
     "What's duck typing?",
     "Caring about what an object can do rather than what it is. If it has the "
     "method you need, it works — no shared base class required. It's why you can "
     "pass anything file-like to code that expects a file, and why interfaces in "
     "Python tend to be conventions rather than declarations.",
     ["behaviour matters more than declared type",
      "no common base class is required",
      "an object is acceptable if it has the needed methods",
      "supports substituting file-like or list-like objects"],
     []),

    (3, "comprehension versus generator expression",
     "What's the difference between a list comprehension and a generator "
     "expression?",
     "The list comprehension builds and holds the whole list. The generator "
     "expression, in round brackets, produces values lazily and holds one at a "
     "time. If you're feeding straight into sum or any, the generator is the "
     "better default — no intermediate list.",
     ["a list comprehension materialises the whole list",
      "a generator expression yields lazily",
      "the generator uses far less memory",
      "the generator can only be consumed once"],
     []),

    (3, "making a class iterable",
     "How would you make your own class work in a for loop?",
     "Implement dunder iter and return an iterator — often the simplest way is "
     "to make dunder iter a generator function and yield from inside it. If you "
     "return self, then the class also needs dunder next and has to raise stop "
     "iteration when it's done.",
     ["implement dunder iter",
      "dunder iter must return an iterator",
      "an iterator needs dunder next",
      "raise StopIteration when exhausted"],
     []),

    (3, "method resolution order",
     "What is method resolution order and when does it start to matter?",
     "It's the linear order Python searches base classes for an attribute. With "
     "single inheritance it's just the chain upward. It matters under multiple "
     "inheritance, where the C3 linearisation decides which parent wins, and "
     "where a chain of super calls actually goes.",
     ["the order in which base classes are searched",
      "computed by C3 linearisation",
      "matters under multiple inheritance",
      "determines which parent's method super calls next"],
     []),

    (3, "composition versus inheritance",
     "When would you use composition instead of inheritance?",
     "Whenever the relationship is 'has a' rather than 'is a'. Inheritance ties "
     "you to the parent's whole interface and its future changes, so it's brittle "
     "for reuse. Composition lets you hold the collaborator and expose only what "
     "you actually want.",
     ["inheritance models an is-a relationship",
      "composition models a has-a relationship",
      "inheritance couples you to the parent's interface",
      "composition is more flexible for reuse"],
     []),

    (3, "garbage collection",
     "How does Python decide when to free an object?",
     "Primarily reference counting — when the last reference goes away, the "
     "object is freed immediately. On top of that there's a cyclic collector, "
     "because a group of objects referring to each other keeps its own counts "
     "above zero and would otherwise never be released.",
     ["reference counting frees objects at zero references",
      "freeing is immediate when the count drops to zero",
      "a separate collector handles reference cycles",
      "cycles would otherwise never be freed"],
     []),

    (3, "attribute lookup",
     "What happens when you access an attribute on an object?",
     "Python checks the instance dictionary, then the class, then the classes "
     "above it in method resolution order. Data descriptors on the class take "
     "priority over the instance dictionary. If nothing matches, dunder getattr "
     "gets one last chance before it raises.",
     ["the instance dictionary is checked first",
      "then the class and its bases in MRO order",
      "descriptors on the class can take priority",
      "dunder getattr is the fallback before AttributeError"],
     []),

    (3, "equality and hashing",
     "If you implement dunder eq on a class, what else do you need to think "
     "about?",
     "Dunder hash. Defining equality without it makes your class unhashable, so "
     "instances can't go in a set or be dictionary keys. And whatever fields you "
     "compare on must be the ones you hash on, or equal objects will land in "
     "different buckets.",
     ["defining dunder eq sets dunder hash to None",
      "the class becomes unhashable and unusable in sets or dict keys",
      "hash must be consistent with equality",
      "equal objects must have equal hashes"],
     []),

    # ---------------------------------------------------------------- 4 ---- #
    # Stretch. Expected to be answered partially, and that is fine.

    (4, "asyncio",
     "How does asyncio differ from threading, and when does it win?",
     "Asyncio is cooperative concurrency on a single thread — tasks yield control "
     "at await points, so there's no preemption and no lock contention. It wins "
     "for large numbers of input-output bound tasks, because a task costs a "
     "fraction of what a thread costs. One blocking call stalls everything.",
     ["cooperative scheduling on a single thread",
      "tasks switch only at await points",
      "scales to far more concurrent I/O tasks than threads",
      "a blocking call blocks the entire event loop"],
     ["What happens if you call a blocking library from inside a coroutine?"]),

    (4, "slots",
     "What are dunder slots for?",
     "Declaring the fixed set of attributes a class allows, so instances use a "
     "compact array instead of a per-instance dictionary. That saves meaningful "
     "memory when you have very many small objects. The cost is you can't add "
     "attributes later, and it interacts awkwardly with multiple inheritance.",
     ["replaces the per-instance dictionary",
      "reduces memory use for many small instances",
      "prevents adding attributes not declared",
      "complicates inheritance and dynamic attributes"],
     []),

    (4, "reference cycles",
     "Given reference counting, how can memory still leak in Python?",
     "Objects in a reference cycle keep each other's counts above zero, so "
     "counting alone never frees them — that's what the cyclic collector is for. "
     "It can still be defeated by objects held alive in caches, module-level "
     "state, or by references held from C extensions.",
     ["cycles keep reference counts above zero",
      "the cyclic garbage collector handles most of these",
      "global or cached references keep objects alive indefinitely",
      "C extension references are outside Python's view"],
     ["What's a weak reference and where would it help here?"]),

    (4, "coroutines versus generators",
     "What's the relationship between a coroutine and a generator?",
     "Both are resumable functions that suspend and keep their state — coroutines "
     "grew out of the generator machinery. The difference is what they suspend "
     "for: a generator yields a value to its consumer, while a coroutine awaits "
     "and hands control back to an event loop.",
     ["both suspend and resume while keeping local state",
      "coroutines were built on the generator protocol",
      "a generator yields values to a consumer",
      "a coroutine awaits and returns control to the event loop"],
     []),

    (4, "the GIL and thread safety",
     "If the GIL means only one thread runs at a time, why do you still need "
     "locks?",
     "Because the lock is released between bytecodes, not held for whole "
     "operations. Anything that reads, modifies and writes back — an increment, a "
     "check-then-act on a dictionary — can be interrupted partway, so two threads "
     "interleave and one update is lost.",
     ["the GIL is released between bytecode instructions",
      "a multi-step operation is not atomic",
      "read-modify-write sequences can interleave",
      "check-then-act patterns still race"],
     []),

    (4, "metaclasses",
     "What's a metaclass, and when would you genuinely need one?",
     "A metaclass is the class of a class — it controls what happens when the "
     "class itself is created. You'd reach for one to register subclasses "
     "automatically, or to validate or rewrite a class definition. Most of the "
     "time a decorator or init subclass is simpler and enough.",
     ["a metaclass is the type of a class",
      "it runs when the class object is created",
      "used for registration or validation of class definitions",
      "usually replaceable by a decorator or init subclass"],
     []),

    (4, "descriptors",
     "What's a descriptor?",
     "An object that defines dunder get, set, or delete and is stored on a class, "
     "so attribute access on instances routes through it. It's the machinery "
     "underneath property, class methods, static methods and slots — anywhere "
     "attribute access needs to run code.",
     ["an object implementing dunder get, set or delete",
      "lives on the class and intercepts attribute access",
      "underlies property, classmethod and staticmethod",
      "data descriptors take priority over the instance dictionary"],
     []),

    (4, "subclassing builtins",
     "What's the catch with subclassing a built-in like dict or list?",
     "The built-in's own methods are implemented in C and mostly don't call back "
     "into your overrides — so overriding set item doesn't change what update or "
     "the constructor do. If you want consistent behaviour you subclass the "
     "collections user dict, or wrap rather than inherit.",
     ["built-in methods do not call your overridden methods",
      "overriding one method leaves others inconsistent",
      "UserDict or UserList behave as expected",
      "composition is often safer than subclassing"],
     []),

    (4, "profiling",
     "You're told a Python function is too slow. How do you approach it?",
     "Measure before changing anything — cProfile for where the time goes by "
     "function, then something line-level on the hot spot. Usually it's an "
     "algorithmic problem or a wrong data structure, not the language. Only after "
     "that would I look at caching, vectorising, or moving work out of Python.",
     ["profile first rather than guessing",
      "named a real tool such as cProfile or timeit",
      "look for algorithmic or data structure problems first",
      "micro-optimise or move to a native library only afterwards"],
     []),

    # ---------------------------------------------------------------- 5 ---- #

    (5, "parallelism in practice",
     "You have a CPU-bound job and eight cores. How do you actually use them "
     "from Python?",
     "Process-based parallelism — a process pool executor, or multiprocessing — "
     "so each worker has its own interpreter and its own lock. Then I'd watch "
     "the serialisation cost, because chunking matters: too fine and you spend "
     "more time pickling than computing. If the work is numeric, a library that "
     "drops the lock in C is often better than any of it.",
     ["use processes rather than threads",
      "each process has an independent interpreter and GIL",
      "serialisation and chunk size dominate the overhead",
      "native libraries that release the GIL are an alternative"],
     []),

    (5, "import system",
     "What actually happens when you import a module, and what can go wrong?",
     "Python finds it on the path, executes it top to bottom once, and caches the "
     "module object so later imports are free. What goes wrong is circular "
     "imports, where a half-initialised module is visible, and shadowing, where a "
     "local file with the same name as a library wins.",
     ["the module is located, executed once, then cached",
      "later imports reuse the cached module",
      "circular imports expose a partially initialised module",
      "a local file can shadow a library of the same name"],
     []),

    (5, "thread-safe cache",
     "How would you design a cache that several threads can safely share?",
     "A dictionary behind a lock is the starting point, but the trap is "
     "check-then-act — two threads both miss and both compute. So the lock has to "
     "cover the miss path too, ideally per key so one slow computation doesn't "
     "block everything. Then eviction, and whether a stale result is acceptable.",
     ["guard the shared structure with a lock",
      "the check-then-act miss path is the real race",
      "per-key locking avoids serialising all callers",
      "considered eviction or bounded size"],
     []),

    (5, "concurrency versus parallelism",
     "In Python specifically, what's the difference between concurrency and "
     "parallelism?",
     "Concurrency is structuring work so it can be interleaved; parallelism is "
     "actually running it at the same time. Threads and asyncio give you "
     "concurrency but, because of the lock, not parallelism for Python-level "
     "work. Only separate processes, or native code that releases the lock, gives "
     "you parallelism.",
     ["concurrency is interleaving, parallelism is simultaneous execution",
      "threads and asyncio provide concurrency only",
      "the GIL prevents parallel bytecode execution",
      "processes or native code give real parallelism"],
     []),

    (5, "designing for testability",
     "How do you write Python that's easy to test?",
     "Push side effects to the edges and keep the core as functions that take "
     "values and return values. Inject dependencies rather than importing them "
     "deep inside, so a test can substitute one without patching module globals. "
     "Heavy mocking is usually a signal the design is wrong, not the test.",
     ["separate pure logic from side effects",
      "inject dependencies instead of hard-coding them",
      "avoid reliance on global or module-level state",
      "heavy patching indicates a design problem"],
     []),
]



def main() -> int:
    # Validation lives in bank_kit, shared with the Applied Science banks, so
    # both are held to the same rules rather than each drifting its own way.
    questions = kit.build("python", Mode.FACTUAL, BANK_DATA)
    kit.check_no_cross_topic_duplicates({"python": questions})
    path = kit.write("python", questions)
    kit.report("python", questions)
    print(f"    -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
