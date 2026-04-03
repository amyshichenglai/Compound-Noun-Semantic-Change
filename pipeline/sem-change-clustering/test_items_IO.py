from typing import List, Tuple, Literal, Dict
from collections import defaultdict
import csv
import os

CCOHA_ERA_TOKEN_COUNTS = {
    (1830, 1859): 44800000,
    (1860, 1889): 54400000,
    (1890, 1919): 63800000,
    (1920, 1949): 72900000,
    (1950, 1979): 71000000,
    (1980, 2009): 81000000,
}
DATASET_TO_ERA_TOKEN_COUNTS = {
    'COHA': CCOHA_ERA_TOKEN_COUNTS
}

DEBUG = False

# TODO: rename to just 'Compound' - since that's how it's often used
class CompositionalityRating:
    # TODO: provide unified (fat) interface for the two kinds of ratings
    #       like e.g. getting a string for the head, regardless of whether it's really
    #       only a lemmatized head or not
    def __init__(self, compound, head, mod):
        self.compound = compound
        self.head = head
        self.mod = mod

    @property
    def primary_component(self) -> str:
        """unified way to access the main target item"""
        return self.compound

    @staticmethod
    def most_compositional_rating() -> float:
        """
        returns top of scale (most compositional) for this rating type
        :return: float
        """
        raise NotImplementedError

    @staticmethod
    def least_compositional_rating() -> float:
        """
        returns bottom of scale (least compositional) for this rating type
        :return: float
        """
        raise NotImplementedError

    def compound_rating(self) -> float:
        raise NotImplementedError

    @property
    def hyphenated_compound(self) -> str:
        return f"{self.mod}-{self.head}"

    def to_list_all(self):
        return [self.compound, self.hyphenated_compound, self.mod, self.head]

    def to_list(self):
        return [self.compound, self.hyphenated_compound]

    def to_list_with_mod(self):
        return [self.compound, self.hyphenated_compound, self.mod]

    def to_list_with_head(self):
        return [self.compound, self.hyphenated_compound, self.head]

    def expand_to_list(self, constituent: Literal["head", "mod", "both"]):
        if constituent == "head":
            return [self.compound, self.hyphenated_compound, self.head]
        elif constituent == "mod":
            return [self.compound, self.hyphenated_compound, self.mod]
        elif constituent == "both":
            return [self.compound, self.hyphenated_compound, self.mod, self.head]
        else:
            raise ValueError("invalid constituent")

    def __eq__(self, other):
        if hasattr(other, 'compound'):
            return self.compound == other.compound and \
            self.mod == other.mod and \
            self.head == other.head
        else:
            return self.compound == other

    def __hash__(self):
        return hash((self.compound, self.mod, self.head))

class TestCompounds:
    def __init__(self, ratings: List[CompositionalityRating]):
        self.compounds = ratings

    @classmethod
    def load_cordeiro(cls, filepath: str):
        items = []
        with open(filepath) as in_f:
            reader = csv.DictReader(in_f, fieldnames=['Compound', 'ModAvg', 'ModStd', 'HeadAvg' ,'HeadStd',
                                                      'CpdAvg','CpdStd','Dataset'],
                delimiter='\t'
            )
            for i, row in enumerate(reader):
                if i == 0:
                    if len(row) == 8 and row["Compound"] == "Compound" and \
                            row["ModAvg"] == "ModAvg":
                        continue
                modifier, head = row["Compound"].split()
                items.append(CompositionalityRatingCordeiro(
                    compound=row['Compound'],
                    head=head,
                    mod=modifier,
                    avg_head=float(row['HeadAvg']),
                    avg_mod=float(row['ModAvg']),
                    compositionality=float(row['CpdAvg']),
                ))
        return cls(items)

    @classmethod
    def load_ghost(cls, filepath: str):
        comp_ratings = []
        with open(filepath, encoding='utf-8') as in_f:
            for i, line in enumerate(in_f):
                if i == 0:
                    # double check that the line is the metadata one
                    fields = line.split('\t')
                    if len(fields) == 7 and fields[0] == "Compound" and fields[1] == "Modifier":
                        continue
                comp_ratings.append(CompositionalityRatingGhost(*line.strip().split('\t')))
        return cls(comp_ratings)

    def to_list(self):
        return self.compounds

    def least_compositional(self, criteria: Literal['head', 'mod', 'both', 'compound']):
        """
       return only the (approximately) lower third of the compositionality scale
       with respect to the head or modifier constituent alone, only if both fulfill
       criteria, or if the overall compound rating (if available) fulfills the criteria.
        """
        rating_type = type(self.compounds[0])
        assert all(type(c) == rating_type for c in self.compounds)
        lowest, highest = rating_type.least_compositional_rating(), rating_type.most_compositional_rating()
        total_range = highest - lowest
        cutoff = lowest + (total_range / 3)
        return [c for c in self.compounds if _compositionality_threshold(
            rating=c, threshold=cutoff, less_than=True, criteria=criteria
        )]

    def most_compositional(self, criteria: Literal['head', 'mod', 'both', 'compound']):
        """
        return only the (approximately) upper third of the compositionality scale
        with respect to the head or modifier constituent alone, only if both fulfill
        criteria, or if the overall compound rating (if available) fulfills the criteria.
        :return:
        """
        rating_type = type(self.compounds[0])
        assert all(type(c) == rating_type for c in self.compounds)
        lowest, highest = rating_type.least_compositional_rating(), rating_type.most_compositional_rating()
        total_range = highest - lowest
        cutoff = highest - (total_range / 3)
        return [c for c in self.compounds if _compositionality_threshold(
            rating=c, threshold=cutoff, less_than=False, criteria=criteria,
        )]

def _compositionality_threshold(
        rating: CompositionalityRating,
        threshold: float,
        less_than: bool,
        criteria: Literal['head', 'mod', 'both', 'compound']
) -> bool:
    comp_attr = None
    if criteria == 'head':
        comp_attr = ["mean_head_rating"]
    elif criteria == 'mod':
        comp_attr = ["mean_mod_rating"]
    elif criteria == 'both':
        comp_attr = ["mean_head_rating", "mean_mod_rating"]
    elif criteria == 'compound':
        comp_attr = ["compound_rating"]
    if not comp_attr:
        raise ValueError("invalid compositionality rating type")
    if less_than:
        return all(getattr(rating, attr) < threshold for attr in comp_attr)
    return all(getattr(rating, attr) > threshold for attr in comp_attr)


class TestRelatedCompounds:
    MINIMUM_COUNT = 10
    MINIMUM_SET_SIZE = 2
    def __init__(self, related_compound_lookup: Dict["CompositionalityRating", "RelatedCompoundSet"]):
        self.compositionality_rated_compound_to_related_compounds = related_compound_lookup

    @classmethod
    def load(
            cls,
            *,
            comp_rated_compounds: List["CompositionalityRating"],
            related_compounds_filename: str,
            constituent_type: Literal["mod", "head"],
            lang: Literal["en", "de"],
            cached_vecs_counts=None, # can only exist *after* vecs have been cached in the first place
    ):
        # TODO: there are differences between the counts
        #       in the related_compounds file
        #       and the counts obtained by caching examples
        #       of these related compounds...
        #       This needs to be resolved for the sampling weights
        #       to be coherent later on

        related_compounds_lookup = TestRelatedCompounds._load_related_compounds_file(related_compounds_filename, lang)
        if DEBUG:
            print(f"related compounds lookup: {related_compounds_lookup}")
            print(f"are we running w/o an existing cache?: {cached_vecs_counts}")
        if cached_vecs_counts is None:
            this_lookup = {}
            for comp_rating in comp_rated_compounds:
                constituent = getattr(comp_rating, constituent_type)
                if constituent not in related_compounds_lookup:
                    continue
                this_lookup[comp_rating] = RelatedCompoundSet(
                    related_compounds_lookup[constituent],
                    constituent_type=constituent_type,
                    shared_constituent=getattr(comp_rating, constituent_type),
                )
            return cls(this_lookup)
        this_lookup = {}
        for comp_rating in comp_rated_compounds:
            if not all([inner_d[comp_rating.primary_component] >= TestRelatedCompounds.MINIMUM_COUNT
                        for inner_d in cached_vecs_counts.values()]):
                continue
            constituent = getattr(comp_rating, constituent_type)
            if constituent in related_compounds_lookup:
                if cached_vecs_counts:
                    related_compound_list = []
                    for related_compound in related_compounds_lookup[constituent]:
                        skip_this_compound = False
                        for time_slice in cached_vecs_counts:
                            if cached_vecs_counts[time_slice][related_compound] < TestRelatedCompounds.MINIMUM_COUNT:
                                skip_this_compound = True
                        if not skip_this_compound:
                            related_compound_list.append(related_compound)

                    if comp_rating.compound in related_compound_list:
                        related_compound_list.remove(comp_rating.compound)
                    if len(related_compound_list) < TestRelatedCompounds.MINIMUM_SET_SIZE:
                        continue
                    related_compound_list.insert(0, comp_rating.compound)
                else:
                    related_compound_list = [c for c in related_compounds_lookup[constituent]]

                this_lookup[comp_rating] = RelatedCompoundSet(
                    related_compound_list,
                    shared_constituent=constituent,
                    constituent_type=constituent_type,
                )

        return cls(this_lookup)

    @classmethod
    def _load_related_compounds_file(cls, filename, lang: Literal["en", "de"]):
        with open(filename, encoding='utf-8') as in_f:
            shared_constituent_to_compounds = defaultdict(list)
            for line in in_f:
                shared_constituent, compound, early_count, late_count = line.strip().split("\t")
                # early_count, late_count = int(early_count), int(late_count)
                # if early_count < TestRelatedCompounds.MINIMUM_COUNT or late_count < TestRelatedCompounds.MINIMUM_COUNT:
                #     continue
                if lang == "de":
                    shared_constituent = shared_constituent.capitalize()
                shared_constituent_to_compounds[shared_constituent].append(compound)
        return {
            k: v
            for k, v in shared_constituent_to_compounds.items()
            # if len(v) >= TestRelatedCompounds.MINIMUM_SET_SIZE
        }

    @classmethod
    def load_pre_filtered(cls,
        *,
        comp_rated_compounds: List["CompositionalityRating"],
        pre_filtered_related_compounds_list_file: str,
        constituent_type: Literal["mod", "head"],
    ):
        lookup = {}
        with open(pre_filtered_related_compounds_list_file, encoding='utf-8') as in_f:
            for line in in_f:
                compounds = line.strip().split(',')
                base = compounds[0]
                lookup[base] = compounds
        # map strings to CompositionalityRating objects and create RelatedCompoundSet objects
        output = {}
        for comp_rating in comp_rated_compounds:
            if comp_rating.primary_component not in lookup:
                continue
            constituent = getattr(comp_rating, constituent_type)
            output[comp_rating] = RelatedCompoundSet(
                lookup[comp_rating.primary_component],
                shared_constituent=constituent,
                constituent_type=constituent_type,
            )
        return output

    def to_list(self):
        return [
            related_compound_set
            for _, related_compound_set
            in self.compositionality_rated_compound_to_related_compounds.items()
        ]

class TestWord:
    def __init__(self, word: str):
        self.word = word

    @property
    def primary_component(self) -> str:
        return self.word

class TestWords:
    def __init__(self, words: List[TestWord]):
        self.words = words

    @classmethod
    def load_semeval2020_list(cls, filepath: str):
        items = []
        with open(filepath, encoding='utf-8') as in_f:
            for line in in_f:
                line = line.strip()
                if "_" in line:
                    line = line.split("_")[0]
                items.append(TestWord(line))
        return cls(items)

    def to_list(self):
        return self.words


class CompositionalityRatingCordeiro(CompositionalityRating):
    def __init__(self, compound: str,
                head: str, mod: str,
                avg_head: float, avg_mod: float,
                compositionality: float
    ):
        super(CompositionalityRatingCordeiro, self).__init__(
            compound=compound, head=head, mod=mod
        )
        self.mean_head_rating = avg_head
        self.mean_mod_rating = avg_mod
        self.compound_rating = compositionality

    @staticmethod
    def most_compositional_rating() -> float:
        return 5.0

    @staticmethod
    def least_compositional_rating() -> float:
        return 0.0

    def __str__(self):
        return f"{self.compound}: ({self.compound_rating})\n" \
               f"\tmod: {self.mod} ({self.mean_mod_rating})\n" \
               f"\thead: {self.head} ({self.mean_head_rating})"

    def __repr__(self):
        return f"CompositionalityRatingCordeiro(compound='{self.compound}'," \
               f"head='{self.head}'," \
               f"mod='{self.mod}'," \
               f"avg_head={self.mean_head_rating}," \
               f"avg_mod={self.mean_mod_rating}," \
               f"compositionality={self.compound_rating})"

    def __eq__(self, other):
        str_eq = super(CompositionalityRatingCordeiro, self).__eq__(other)
        if hasattr(other, "compound"):
            return str_eq and \
                self.mean_mod_rating == other.mean_mod_rating and \
                self.mean_head_rating == other.mean_head_rating and \
                self.compound_rating == other.compound_rating
        else:
            return self.compound == other

    def __hash__(self):
        return hash(self.compound) ^ hash(self.mod) ^ hash(self.head)

class CompositionalityRatingGhost(CompositionalityRating):
    def __init__(self, compound: str, mod: str, head: str,
            num_ratings_mod: int, num_ratings_head: int,
            mean_mod_rating: str, mean_head_rating: str
    ):
        super(CompositionalityRatingGhost, self).__init__(
            compound=compound, head=head, mod=mod
        )
        self.mean_mod_rating: float = float(mean_mod_rating)
        self.mean_head_rating: float = float(mean_head_rating)

    @staticmethod
    def most_compositional_rating() -> float:
        return 6.0

    @staticmethod
    def least_compositional_rating() -> float:
        return 1.0

    @property
    def hyphenated_compound(self) -> str:
        # check for Fugenelement
        before_head = "".join(self.compound.split(self.head.lower())[:-1])
        after_mod = "".join(before_head.split(self.mod)[1:])
        if after_mod:
            return f"{self.mod}{after_mod}-{self.head}"
        return super().hyphenated_compound

    @property
    def compound_rating(self) -> float:
        return (self.mean_mod_rating + self.mean_head_rating) / 2

    def __str__(self):
        return f"{self.compound}\n" \
               f"\tmod: {self.mod} ({self.mean_mod_rating})\n" \
               f"\thead: {self.head} ({self.mean_head_rating})"

    def __repr__(self):
        return f"CompositionalityRatingGhost(compound='{self.compound}'," \
               f"head='{self.head}'," \
               f"mod='{self.mod}'," \
               f"mean_head_rating={self.mean_head_rating}," \
               f"mean_mod_rating={self.mean_mod_rating}"

    def __eq__(self, other):
        str_eq = super(CompositionalityRatingGhost, self).__eq__(other)
        return str_eq and \
            self.mean_mod_rating == other.mean_mod_rating and \
            self.mean_head_rating == other.mean_head_rating

    def __hash__(self):
        return hash(self.compound) ^ hash(self.mod) ^ hash(self.head)


class RelatedCompoundSet:
    def __init__(self, compounds: List[str], shared_constituent: str, constituent_type: Literal["head", "mod"]):
        self.compounds = compounds
        self.shared_constituent = shared_constituent
        self.constituent_type = constituent_type

    # does this actually make sense? how often do we know in advance
    # that one compound *might* be more interesting than the rest of the set?
    @property
    def primary_component(self):
        assert len(self.compounds)
        return self.compounds[0]

    @property
    def secondary_components(self):
        assert len(self.compounds) > 1
        return self.compounds[1:]
    
    def __str__(self):
        return f"{self.shared_constituent}: " + ", ".join(c for c in self.compounds)

    def __repr__(self):
        return f"RelatedCompoundSet(compounds=[{[','.join(c for c in self.compounds)]}]"\
        f", shared_constituent={self.shared_constituent})"

    def expand_to_list(self, constituent=None):
        return self.compounds
    

class FrequencyStats:
    def __init__(
            self,
            file_paths: List[str],
            dataset: Literal['COHA', 'DTA'],
    ):
        eras = _eras_from_filepaths(file_paths)
        # split up either e.g.  freqs_cordeiro_reddy_1830s-1850s.tsv

        self._lookup = load_freq_stats(file_paths, eras, dataset)


class ProductivityStats:
    def __init__(self, file_paths, dataset: Literal['COHA', 'DTA']):
        eras = _eras_from_filepaths(file_paths)

def _eras_from_filepaths(file_paths: List[str]) -> List[Tuple[int, int]]:
    eras = []
    for filepath in file_paths:
        # get the basename
        base = os.path.basename(filepath)
        # remove extension
        no_ext = base.split('.')[0]
        era_text = no_ext.split('_')[-1]
        era1, era2 = era_text.replace('s', '').split('-')
        eras.append((int(era1), int(era2) + 9))
    return eras

def load_freq_stats(filenames: List[str], eras: List[Tuple[int, int]], dataset: Literal['COHA', 'DTA']):
    """frequency stats in .tsv files"""
    # compound        mod     head    mod_pos head_pos        mag     fic     news    nf      total
    lookup = {} # compound -> era -> freq
    for filename, era in zip(filenames, eras):
        # retrieve token count for era
        era_to_range = (era[0], era[1] + 9)
        tokens_for_era = DATASET_TO_ERA_TOKEN_COUNTS[dataset][era_to_range]
        with open(filename, encoding='utf-8') as in_f:
            reader = csv.DictReader(
                in_f,
                delimiter='\t'
            )
            for i, row in enumerate(reader):
                if row['compound'] not in lookup:
                    lookup[row['compound']] = {}
                lookup[row['compound']][era] = int(row['total']) / tokens_for_era
    return lookup


def load_prod_stats(filenames: List[str], eras: List[Tuple[int, int]], dataset: Literal['COHA', 'DTA']):
    """productivity stats in .tsv files"""
    lookup = {} # lemma -> era -> prod
    for filename, era in zip(filenames, eras):
        era_to_range = (era[0], era[1] + 9)
        tokens_for_era = DATASET_TO_ERA_TOKEN_COUNTS[dataset][era_to_range]
        with open(filename, encoding='utf-8') as in_f:
            reader = csv.DictReader(in_f, fieldnames=['lemma', 'productivity'], delimiter=',')
            for i, row in enumerate(reader):
                if i == 0 and row['lemma'] == 'lemma' and row['productivity'] == 'productivity':
                    continue
                if row['lemma'] not in lookup:
                    lookup[row['lemma']] = {}
                lookup[row['lemma']][era] = float(row['productivity']) / tokens_for_era
    return lookup

if __name__ == "__main__":
    # just for quick testing:
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("compounds_file", help="file w/ compounds, ratings")
    parser.add_argument("file_type", choices=["cordeiro", "ghost"])
    args = parser.parse_args()

    if args.file_type == "cordeiro":
        test_items = TestCompounds.load_cordeiro(args.compounds_file)
        for compound in test_items.compounds:
            print(compound.compound)
    elif args.file_type == "ghost":
        test_items = TestCompounds.load_ghost(args.compounds_file)
        for compound in test_items.compounds:
            print(compound.compound)
