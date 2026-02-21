# Teorie grafů

Tento repozitář obsahuje průběžně rozvíjený a postupně rozšiřovaný výzkumný projekt z oblasti teorie grafů. Projekt je koncipován jako výpočetní a strukturální rámec pro zkoumání různých problémů teorie grafů. Je navržen jako rozšiřitelný a bude postupně doplňován o další směry výzkumu.

Aktuální zaměření

V současnosti jsou hlavní cíle projektu následující:
1. Generování všech graciózních Prüferových kódů.
2. Enumerace všech neoznačených stromů s n vrcholy.
3. Výpočet grup automorfismů těchto stromů.

## Grációzní Prüferovy kódy

Tento repozitář obsahuje zdrojový kód a otevřená data související se studiem grációzních ohodnocení stromů reprezentovaných pomocí Prüferových kódů.

Strom na n vrcholech připouští grációzní ohodnocení, pokud lze jeho vrcholy ohodnotit navzájem různými celými čísly tak, aby absolutní rozdíly ohodnocení na hranách tvořily přesně množinu {1, 2, ..., n−1}. V klasické literatuře jsou grációzní ohodnocení obvykle chápána především jako vlastnost stromu jakožto abstraktního grafu a pozornost je věnována zejména otázkám existence a počtu takových stromů. V tomto projektu je však důraz kladen na jednotlivá grációzní ohodnocení jako na konkrétní kombinatorické objekty.

Základní myšlenkou je propojení grációzních ohodnocení s Prüferovými kódy prostřednictvím Sheppardových kódů. Každý Sheppardův kód určuje graf pomocí pevně dané konstrukce hran. Pokud je výsledný graf stromem, vzniklé ohodnocení je grációzní z definice konstrukce. Takto získané stromy jsou následně převedeny na Prüferovy kódy. Tím vzniká explicitní reprezentace grációzních ohodnocení jako prvků prostoru Prüferových kódů.

Data v tomto repozitáři se skládají právě z těchto Prüferových kódů. Ohodnocení vrcholů je voleno z množiny {0, 1, ..., n−1} a každý řádek datového souboru odpovídá jednomu Prüferovu kódu délky n−2. Každý takový kód tedy reprezentuje konkrétní grációzní ohodnocení.

Část kódu pro práci s grafy vychází z modulu Diktyonphi, který byl původně vytvořen Jiřím Fišerem a Jiřím Škvarou z katedry informatiky UJEP pro výukové účely, a byl dále upraven a rozšířen pro potřeby tohoto projektu. Původní verze: https://github.com/Jiri-Fiser/DiktyonPhi/blob/master/diktyonphi.py

Generování dat je algoritmické a reprodukovatelné. Pro pevně daný počet vrcholů n je enumerován prostor Sheppardových kódů, tyto kódy jsou převedeny na grafy, z nichž jsou vybrány pouze stromy, a ty jsou následně reprezentovány pomocí Prüferových kódů. Celkové počty získané tímto postupem souhlasí s dříve publikovanými výsledky o počtech grációzních stromů, což slouží jako ověření správnosti implementace.

Tato práce propojuje grációzní ohodnocení stromů s Prüferovými kódy a zaměřuje se na vztah mezi těmito dvěma objekty, které byly dosud zkoumány převážně samostatně.

Tento repozitář je součástí bakalářské práce a je zveřejněn za účelem zajištění reprodukovatelnosti a dalšího zkoumání grációzních ohodnocení stromů v reprezentaci pomocí Prüferových kódů.

## Licence

Tento projekt je uvolněn pod licencí MIT.
