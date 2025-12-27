# Datasets of graceful Prüfer codes

Each subdirectory corresponds to a fixed number of vertices \( n \).

Each subdirectory contains one or more files providing a complete list of
Prüfer codes representing individual graceful labelings of trees on
\( n \) vertices, listed in the order in which they were found by the algorithm.
In addition, a subdirectory named `sorted` is provided, containing the same
files with the Prüfer codes sorted lexicographically. In these files, each
code is annotated with its rank in the space of all Prüfer codes for the
given \( n \).

## Data format

- in the sorted files, each line begins with the rank of the code  
- one Prüfer code per line  
- integers separated by spaces  
- vertex labels are taken from the set \( \{0,1,\dots,n-1\} \)  
- the length of each code is \( n-2 \)

Each line represents a specific graceful labeling of a tree.


# Datové sady graciózních Prüferových kódů

Každý podadresář odpovídá pevnému počtu vrcholů \( n \).

Každý podadresář obsahuje jeden či více souborů, které poskytují úplný seznam
Prüferových kódů reprezentujících jednotlivá graciózní ohodnocení stromů
na \( n \) vrcholech v pořadí, ve kterém byly nalezeny algoritmem.
Kromě toho obsahuje také podadresář `sorted`, který obsahuje tytéž soubory,
avšak s Prüferovými kódy lexikograficky seřazenými. V těchto souborech je u
každého kódu uvedeno jeho pořadí v prostoru všech Prüferových kódů
pro dané \( n \).

## Formát dat

- v seřazených souborech je na začátku řádku uvedeno pořadí kódu  
- jeden Prüferův kód na řádek  
- celá čísla oddělená mezerami  
- označení vrcholů je z množiny \( \{0,1,\dots,n-1\} \)  
- délka kódu je \( n-2 \)

Každý řádek reprezentuje konkrétní graciózní ohodnocení stromu.
