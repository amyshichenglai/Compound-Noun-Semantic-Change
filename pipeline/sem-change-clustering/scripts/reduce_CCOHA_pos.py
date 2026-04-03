# link to tagset and descriptions: https://ucrel.lancs.ac.uk/claws7tags.html

from argparse import ArgumentParser
import sys

MAPPING = {
    "APPGE": "PN", # 	possessive pronoun, pre-nominal (e.g. my, your, our)
    "AT": "DD", # 	article (e.g. the, no)
    "AT1": "DD", # 	singular article (e.g. a, an, every)
    "BCL": "CS", # 	before-clause marker (e.g. in order (that),in order (to))
    "CC": "CC", # 	coordinating conjunction (e.g. and, or)
    "CCB": "CC",# 	adversative coordinating conjunction ( but)
    "CS": "CS",# 	subordinating conjunction (e.g. if, because, unless, so, for)
    "CSA": "CS", # 	as (as conjunction)
    "CSN": "CS", # 	than (as conjunction)
    "CST": "CS", # 	that (as conjunction)
    "CSW": "CS", # 	whether (as conjunction)
    "DA": "DD", # 	after-determiner or post-determiner capable of pronominal function (e.g. such, former, same)
    "DA1": "DD", # 	singular after-determiner (e.g. little, much)
    "DA2": "DD", # 	plural after-determiner (e.g. few, several, many)
    "DAR": "DD", # 	comparative after-determiner (e.g. more, less, fewer)
    "DAT": "DD", # 	superlative after-determiner (e.g. most, least, fewest)
    "DB": "DD", # 	before determiner or pre-determiner capable of pronominal function (all, half)
    "DB2": "DD", # 	plural before-determiner ( both)
    "DD": "DD", # 	determiner (capable of pronominal function) (e.g any, some)
    "DD1": "DD", # 	singular determiner (e.g. this, that, another)
    "DD2": "DD", # 	plural determiner ( these,those)
    "DDQ": "DD", # 	wh-determiner (which, what)
    "DDQGE": "DD", # 	wh-determiner, genitive (whose)
    "DDQV": "DD", # 	wh-ever determiner, (whichever, whatever)
    "EX": "EX", # 	existential there
    "FO": "FO", # 	formula
    "FU": "FU", # 	unclassified word
    "FW": "FW", # 	foreign word
    "GE": "GE", # 	germanic genitive marker - (' or's)
    "IF": "II", # 	for (as preposition)
    "II": "II", # 	general preposition
    "IO": "II", # 	of (as preposition)
    "IW": "II", # 	with, without (as prepositions)
    "JJ": "JJ", # 	general adjective
    "JJR": "JJ", # 	general comparative adjective (e.g. older, better, stronger)
    "JJT": "JJ", # 	general superlative adjective (e.g. oldest, best, strongest)
    "JK": "JJ", # 	catenative adjective (able in be able to, willing in be willing to)
    "MC": "MM", # 	cardinal number,neutral for number (two, three..)
    "MC1": "MM", # 	singular cardinal number (one)
    "MC2": "MM", # 	plural cardinal number (e.g. sixes, sevens)
    "MCGE": "MM", # 	genitive cardinal number, neutral for number (two's, 100's)
    "MCMC": "MM", # 	hyphenated number (40-50, 1770-1827)
    "MD": "MM", # 	ordinal number (e.g. first, second, next, last)
    "MF": "MM", # 	fraction,neutral for number (e.g. quarters, two-thirds)
    "ND1": "NN", # 	singular noun of direction (e.g. north, southeast)
    "NN": "NN", # 	common noun, neutral for number (e.g. sheep, cod, headquarters)
    "NN1": "NN", # 	singular common noun (e.g. book, girl)
    "NN2": "NN", # 	plural common noun (e.g. books, girls)
    "NNA": "NN", # 	following noun of title (e.g. M.A.)
    "NNB": "NN", # 	preceding noun of title (e.g. Mr., Prof.)
    "NNL1": "NN", # 	singular locative noun (e.g. Island, Street)
    "NNL2": "NN", # 	plural locative noun (e.g. Islands, Streets)
    "NNO": "NN", # 	numeral noun, neutral for number (e.g. dozen, hundred)
    "NNO2": "NN", # 	numeral noun, plural (e.g. hundreds, thousands)
    "NNT1": "NN", # 	temporal noun, singular (e.g. day, week, year)
    "NNT2": "NN", # 	temporal noun, plural (e.g. days, weeks, years)
    "NNU": "NN", # 	unit of measurement, neutral for number (e.g. in, cc)
    "NNU1": "NN", # 	singular unit of measurement (e.g. inch, centimetre)
    "NNU2": "NN", # 	plural unit of measurement (e.g. ins., feet)
    "NP": "NP", # 	proper noun, neutral for number (e.g. IBM, Andes)
    "NP1": "NP", # 	singular proper noun (e.g. London, Jane, Frederick)
    "NP2": "NP", # 	plural proper noun (e.g. Browns, Reagans, Koreas)
    "NPD1": "NN", # 	singular weekday noun (e.g. Sunday)
    "NPD2": "NN", # 	plural weekday noun (e.g. Sundays)
    "NPM1": "NN", # 	singular month noun (e.g. October)
    "NPM2": "NN", # 	plural month noun (e.g. Octobers)
    "PN": "PN", # 	indefinite pronoun, neutral for number (none)
    "PN1": "PN", # 	indefinite pronoun, singular (e.g. anyone, everything, nobody, one)
    "PNQO": "PN", # 	objective wh-pronoun (whom)
    "PNQS": "PN", # 	subjective wh-pronoun (who)
    "PNQV": "PN", # 	wh-ever pronoun (whoever)
    "PNX1": "PN", # 	reflexive indefinite pronoun (oneself)
    "PPGE": "PN", # 	nominal possessive personal pronoun (e.g. mine, yours)
    "PPH1": "PN", # 	3rd person sing. neuter personal pronoun (it)
    "PPHO1": "PN", # 	3rd person sing. objective personal pronoun (him, her)
    "PPHO2": "PN", # 	3rd person plural objective personal pronoun (them)
    "PPHS1": "PN", # 	3rd person sing. subjective personal pronoun (he, she)
    "PPHS2": "PN", # 	3rd person plural subjective personal pronoun (they)
    "PPIO1": "PN", # 	1st person sing. objective personal pronoun (me)
    "PPIO2": "PN", # 	1st person plural objective personal pronoun (us)
    "PPIS1": "PN", # 	1st person sing. subjective personal pronoun (I)
    "PPIS2": "PN", # 	1st person plural subjective personal pronoun (we)
    "PPX1": "PN", # 	singular reflexive personal pronoun (e.g. yourself, itself)
    "PPX2": "PN", # 	plural reflexive personal pronoun (e.g. yourselves, themselves)
    "PPY": "PN", # 	2nd person personal pronoun (you)
    "RA": "RR", # 	adverb, after nominal head (e.g. else, galore)
    "REX": "RR", # 	adverb introducing appositional constructions (namely, e.g.)
    "RG": "RR", # 	degree adverb (very, so, too)
    "RGQ": "RR", # 	wh- degree adverb (how)
    "RGQV": "RR", # 	wh-ever degree adverb (however)
    "RGR": "RR", # 	comparative degree adverb (more, less)
    "RGT": "RR", # 	superlative degree adverb (most, least)
    "RL": "RR", # 	locative adverb (e.g. alongside, forward)
    "RP": "RR", # 	prep. adverb, particle (e.g about, in)
    "RPK": "RR", # 	prep. adv., catenative (about in be about to)
    "RR": "RR", # 	general adverb
    "RRQ": "RR", # 	wh- general adverb (where, when, why, how)
    "RRQV": "RR", # 	wh-ever general adverb (wherever, whenever)
    "RRR": "RR", # 	comparative general adverb (e.g. better, longer)
    "RRT": "RR", # 	superlative general adverb (e.g. best, longest)
    "RT": "RR", # 	quasi-nominal adverb of time (e.g. now, tomorrow)
    "TO": "II", # 	infinitive marker (to)
    "UH": "UH", # 	interjection (e.g. oh, yes, um)
    "VB0": "VA", # 	be, base form (finite i.e. imperative, subjunctive)
    "VBDR": "VA", # 	were
    "VBDZ": "VA", # 	was
    "VBG": "VA", # 	being
    "VBI": "VA", # 	be, infinitive (To be or not... It will be ..)
    "VBM": "VA", # 	am
    "VBN": "VA", # 	been
    "VBR": "VA", # 	are
    "VBZ": "VA", # 	is
    "VD0": "VA", # 	do, base form (finite)
    "VDD": "VA", # 	did
    "VDG": "VA", # 	doing
    "VDI": "VA", # 	do, infinitive (I may do... To do...)
    "VDN": "VA", # 	done
    "VDZ": "VA", # 	does
    "VH0": "VA", # 	have, base form (finite)
    "VHD": "VA", # 	had (past tense)
    "VHG": "VA", # 	having
    "VHI": "VA", # 	have, infinitive
    "VHN": "VA", # 	had (past participle)
    "VHZ": "VA", # 	has
    "VM": "VM", # 	modal auxiliary (can, will, would, etc.)
    "VMK": "VM", # 	modal catenative (ought, used)
    "VV0": "VV", # 	base form of lexical verb (e.g. give, work)
    "VVD": "VV", # 	past tense of lexical verb (e.g. gave, worked)
    "VVG": "VV", # 	-ing participle of lexical verb (e.g. giving, working)
    "VVGK": "VV", # 	-ing participle catenative (going in be going to)
    "VVI": "VV", # 	infinitive (e.g. to give... It will work...)
    "VVN": "VV", # 	past participle of lexical verb (e.g. given, worked)
    "VVNK": "VV", # 	past participle catenative (e.g. bound in be bound to)
    "VVZ": "VV", # 	-s form of lexical verb (e.g. gives, works)
    "XX": "XX", # 	not, n't
    "ZZ1": "NN", # 	singular letter of the alphabet (e.g. A,b)
    "ZZ2": "NN", # 	plural letter of the alphabet (e.g. A's, b's)
    # our additions:
    "Y": "Y", # PUNCTUATION
    "<EOS>": "<EOS>",
    "<NO-SEQ>": "<NO-SEQ>",
    "!": "!",
    "Q!": "Q!",
}

def reduce_tag(tag: str) -> str:
    single_tag = None
    # multiple possible tags are concatenated with "_"

    # remove the CCOHA-created _<sub> additions to tags:
    tags = tag.replace("_<sub>", "")

    tag_splits = tag.split("_")
    # tags may be suffixed with "@" if not very likely, or "%" if extremely unlikely
    # we try to take the first non-unlikely-suffixed tag
    if all([(t.endswith("@") or t.endswith("%") for t in tag_splits)]):
        single_tag = tag_splits[0]
    else:
        for t in tag_splits:
            if not (t.endswith("@") or t.endswith("%")):
                single_tag = t
                break
    assert single_tag is not None
    single_tag = single_tag.replace("@", "").replace("%", "").strip()
    # if the last two chars are both numbers, these are DITTO tags - saying that a 
    # multi-word expression counts as one POS-tagged entity - e.g. the adverbial phrase
    # "at length" could be tagged at_RR21 length_RR22, where the 1st digit identifies how 
    # many tokens the expression has, and the 2nd digit indexes (starting at 1) the token
    if single_tag[-2:].isnumeric():
        single_tag = single_tag[:-2]
    # here we just remove the ditto tag if present. 

    # now we should have a string that can be translated through our mapping
    if single_tag.upper() not in MAPPING:
        return single_tag.upper()
    return MAPPING[single_tag.upper()]



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("in_file")
    args = parser.parse_args()
    with open(args.in_file, encoding='utf-8') as in_f:
        for line in in_f:
            tok, lem, tag = line.split("\t")
            print(f"{tok}\t{lem}\t{reduce_tag(tag).lower()}")
