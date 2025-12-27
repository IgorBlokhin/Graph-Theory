# Grációzní Prüferovy kódy

Tento repozitář obsahuje zdrojový kód a otevřená data související se studiem grációzních ohodnocení stromů reprezentovaných pomocí Prüferových kódů.

Strom na n vrcholech připouští grációzní ohodnocení, pokud lze jeho vrcholy ohodnotit navzájem různými celými čísly tak, aby absolutní rozdíly ohodnocení na hranách tvořily přesně množinu {1, 2, ..., n−1}. V klasické literatuře jsou grációzní ohodnocení obvykle chápána především jako vlastnost stromu jakožto abstraktního grafu a pozornost je věnována zejména otázkám existence a počtu takových stromů. V tomto projektu je však důraz kladen na jednotlivá grációzní ohodnocení jako na konkrétní kombinatorické objekty.

Základní myšlenkou je propojení grációzních ohodnocení s Prüferovými kódy prostřednictvím Sheppardových kódů. Každý Sheppardův kód určuje graf pomocí pevně dané konstrukce hran. Pokud je výsledný graf stromem, vzniklé ohodnocení je grációzní z definice konstrukce. Takto získané stromy jsou následně převedeny na Prüferovy kódy. Tím vzniká explicitní reprezentace grációzních ohodnocení jako prvků prostoru Prüferových kódů.

Data v tomto repozitáři se skládají právě z těchto Prüferových kódů. Ohodnocení vrcholů je voleno z množiny {0, 1, ..., n−1} a každý řádek datového souboru odpovídá jednomu Prüferovu kódu délky n−2. Každý takový kód tedy reprezentuje konkrétní grációzní ohodnocení, nikoli pouze neoznačený strom.

Generování dat je algoritmické a reprodukovatelné. Pro pevně daný počet vrcholů n je enumerována předepsaná část prostoru Sheppardových kódů, tyto kódy jsou převedeny na grafy, z nichž jsou vybrány pouze stromy, a ty jsou následně reprezentovány pomocí Prüferových kódů. Celkové počty získané tímto postupem souhlasí s dříve publikovanými výsledky o počtech grációzních stromů, což slouží jako ověření správnosti implementace.

Smyslem poskytnutí explicitních Prüferových kódů je umožnit práci s jednotlivými grációzními ohodnoceními ve standardní a dobře známé reprezentaci. Tím je možné tato ohodnocení analyzovat stejnými prostředky jako obecné označené stromy v prostoru Prüferových kódů.

Tento repozitář je součástí bakalářské práce a je zveřejněn za účelem zajištění reprodukovatelnosti a dalšího zkoumání grációzních ohodnocení stromů v reprezentaci pomocí Prüferových kódů.

## Licence

Tento projekt je uvolněn pod licencí MIT.
