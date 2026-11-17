"""The rules of the AI-tells scanner (tic_scan.py).

Each rule is a dict:
    level  "hard"  = a tell nearly every time it turns up; the engine's gate rejects a candidate that has one
           "check" = could be a tell, could be a literal use or a real voice; the gate passes it and the judge
                     sees the tag
    label  the short name that goes in a tag: [hard: label] or [check: label]
    regex  matched with case ignored, unless "case": True
    scope  "all"       = anywhere
           "narration" = outside double quotes and outside *italic* spans
           "dialogue"  = inside double quotes only
    stack  True for rules that match a whole sentence (triples, negation frames). They only compete with each
           other for a span, so a word-level hit inside a triple doesn't hide the triple

The banned words come from the house rules file (house_rules.py, rules.toml), so the scanner and the engine's
gates share one list. The stock phrases come from phrases.txt next to this file and go last, so a rule above that
already tags a span wins.

Every label belongs to a family (FAMILIES), grouped by the kind of judgment it needs. The engine's make_tics runs
every rule whose label is not in HOUSE_ONLY, which is empty here: the house-only rules of the scanner this was
taken from stayed behind.
"""
import os
import re

try:
    from . import house_rules as _house
except ImportError:          # loaded from its folder on sys.path, as engine/draws.py make_tics does it
    import house_rules as _house

W = r"[a-z'-]+"          # one word
PRON = r"(?:her|his|my|their|your)"
SUBJ = r"(?:she|he|i|they|you)"

# Clause pieces for the sentence-level rules. A clause starts the line or follows , . ! ? ; a spaced dash or a
# closing quote, and never crosses a quote mark.
CL_START = r"(?:^|(?<=[.!?,;] )|(?<= - )|(?<=[\"”] ))"
CL = r"[^.!?,;\"“”]"
CL_END = r"(?:[.!?,;]+| -)(?: |$)"
LEAD = r"(?:(?:and|or|then|so|but) )?"
NEG = r"(?:not|no|never|nothing|nobody|none|without)"
# "but she..." opens a real second clause, so only a bare "but X" counts as the pivot
BUT_PIVOT = r"but\b(?! (?:i|you|he|she|it|we|they|there|that|this)\b)"
PIVOT = rf"(?:just|only|simply|merely|instead|{BUT_PIVOT})\b"
# openers too common to count alone; a repeat needs the next word too
FUNC = r"(?:the|a|an|her|his|my|their|your|its|our|to|of|in|on|at|she|he|i|it|they|we|you|was|were|is)"
# Adjectives for the adjective-stack rule. There is no tagger, so it's a word list plus safe suffixes. Words that
# double as adverbs or verbs (still, even, close, open, light) stay out.
ADJ_WORDS = """
abrupt aching amber anxious ashamed assured bare beautiful big bitter blunt boneless breathy brief bright
broken brutal calculated calm careful certain clean clinical cocky cold comfortable composed confident
constant content controlled cool crisp crude curvy cute dark dazed deep deliberate delicate delighted dense
desperate dirty dizzy drenched dripping drowsy dumb eager efficient embarrassed empty endless exhausted
exposed familiar fast fat feral fierce filthy firm flushed flustered fluttering fond fragile frantic frenzied
fresh full gentle genuine giddy glossy glowing golden gorgeous gradual grave greedy guilty happy hard hazy
heated heavy helpless high hoarse hollow honest hot huge hungry husky icy immaculate innocent insistent
intense intent intimate inviting kind knowing lazy lean level lewd limp liquid lithe lonely long loose loud
low lush measured mechanical messy methodical miserable molten muscular naked narrow needy nervous numb
obscene overwhelming pale patient perfect perky petite pillowy pink playful pliant plump polished possessive
practiced precise pretty primal private proud puffy pulsing quick quiet ragged rapid raw red relentless
restrained reverent rhythmic rich rigid ripe rough round ruined sad salty satisfied serious settled sexy shaky
sharp sheepish shiny short shuddering shy silent silky silly sincere slack sleepy slender slick slippery
sloppy slow small smooth smug soaked soaking soft solemn sore spent startled steady sticky stiff stupid
stuttering sudden supple sure surprised sweaty sweet swollen syrupy taut teasing tender tense terrifying thick
throbbing tight tiny tired toned trembling triumphant trusting unbothered uneven unfiltered unguarded
unhurried unsteady unwilling urgent violent vulgar warm watery wet wicked wide wild worshipful wrecked young
""".split()
ADJ = rf"(?:{'|'.join(ADJ_WORDS)}|[a-z]{{3,}}(?:ous|ful|less|ive|ish|able|ible))"
ADV = (r"(?!(?:only|really|early|family|belly|reply|supply|ally|apply|rely|holy|ugly|silly|lovely|likely|daily"
       r"|friendly|lonely|jelly|rally|bully|curly|chilly|oily|hilly|melancholy)\b)[a-z]{3,}ly")
OPENER = rf"((?:{FUNC}) [\w'’-]+|(?!(?:{FUNC}|and|or|but|then|so)\b)[\w'’-]{{2,}})\b"
# short noun lists can repeat a single article, possessive or preposition
DET = r"(?:the|a|an|her|his|my|their|your|its|our|to|of|in|on|at|with|from|for|into)"
# The dash-append rules: a description tacked on after a dash that opens on a participle, a possessive or an
# adjective. A spoken aside opens on anything else, so it passes.
DASHES = "\u2014\u2013"
DASH_FIRST = rf"(?:[a-z]{{3,}}(?:ing|ed)|the|her|his|my|your|their|its|our|a|an|{ADJ})\b"
DASH_NOT = r"(?!(?:who|which|because|so|but|like|since|while|except)\b)"
# first words that look like a verb but aren't one, for the subjectless-fragment rule
SUBJLESS_NOT = "|".join("""
Across Afterwards Always Anything Arms Bed Being Besides Blankets Bless Bodies Boys Breasts Building Ceiling
Chess Clothes Clothing Darling Days Does Downstairs Dozens During Ending Evening Everything Eyes Fans Feeling
Fingers Girls Guess Guests Hands Has Hers Hips His Hours Hundreds Indeed Indoors Its King Knees Lashes Legs
Less Lightning Lights Lips Members Minutes Months Morning Nails Naked Need News Nothing Others Ours Outdoors
People Perhaps Pillows Plus Pudding Red Ring Sacred Seconds Seed Sheets Shoulders Something Sometimes Speed
Spring Stairs Streets String Sucks Taxis Tears Thanks Thanksgiving Theirs Thighs Thing Things This Thousands
Thus Tired Towards Unless Upstairs Was Wedding Weeks Wicked Wing Words Years Yes Yours
""".split())


def _banned_word_rules():
    """The house rules file's banned words as two rules: the hard list and the check list."""
    rules = _house.load_rules()["house"]
    out = []
    for key, level, label in (("banned_words", "hard", "banned word"),
                              ("banned_words_check", "check", "banned word (literal?)")):
        pat = _house.word_pattern(rules[key])
        if pat:
            out.append(dict(level=level, label=label, regex=pat, scope="all"))
    return out


RULES = [
    # ---------------------------------------------------------------- dashes, banned words, numbers
    dict(level="hard", label="em dash", regex=r"[\u2014\u2013]", scope="all"),
    *_banned_word_rules(),
    dict(level="check", label="4/7 number",
         regex=r"\b(?:\d*[47]\d*(?:[.,:]\d+)?(?:st|nd|rd|th)?|(?:four|seven)(?:teen|ty|s)?(?:-\w+)?|fourth|seventh|fortieth|seventieth|forty|seventy)\b|\bon all fours\b",
         scope="all"),
    dict(level="check", label="made-up count",
         regex=r"\b(?:\d{2,}|(?:twenty|thirty|fifty|sixty|eighty|ninety|hundred|thousand)\w*)\s+(?:seconds?|minutes?|hours?|times|breaths?|heartbeats?|beats?|steps?|people|men|guys|girls)\b",
         scope="narration"),

    # ---------------------------------------------------------------- stock AI phrases
    # The best-known machine phrases, hard: a candidate with one is out.
    dict(level="hard", label="stock AI phrase",
         regex=r"\b(?:delv(?:e|es|ed|ing)|tapestry|a testament to|in the realm of|it's important to note|navigat\w+ the complexit\w+|"
               r"a symphony of|a dance of|unspoken (?:bond|understanding)|sent shivers down)\b",
         scope="all"),
    dict(level="check", label="sums up its own point",
         regex=r"\b(?:and that's|which is) (?:the|what) (?:point|beauty)\b", scope="all"),

    # ---------------------------------------------------------------- tags and the voice
    dict(level="hard", label="voice adjective tag",
         regex=rf"\bin an? (?:{W} )?(?:low|soft|small|quiet|husky|breathy|hoarse|rough|sultry|teasing|playful|sing-?song|flat|cold|icy|gentle|tight|strained|thick|shaky|smoky|silky|honeyed|deep) (?:voice|tone|whisper)\b",
         scope="narration"),
    dict(level="check", label="voice described",
         regex=rf"\b(?:{PRON}|[A-Z][a-z]+'s) (?:voice|tone)\b(?! (?:memo|note|message|actor|acting|call|chat|coach|lesson|of the (?:group|book)))",
         scope="narration"),
    dict(level="check", label="named emotion",
         regex=r"\b(?:with|in) (?:a (?:mix(?:ture)?|blend) of |barely (?:concealed|contained|suppressed) |obvious |open |mock |growing |sudden )?(?:irritation|exasperation|frustration|amusement|annoyance|anticipation|desire|longing|relief|guilt|shame|embarrassment|confusion|determination|disbelief|horror|panic|excitement|satisfaction|disappointment|resignation|triumph|mischief|curiosity|concern)\b|\ba (?:mix(?:ture)?|blend) of \w+ and \w+",
         scope="narration"),
    # "managed" as a speech tag hands the talk over as a struggle seen from outside the speaker
    dict(level="check", label="managed as a tag",
         regex=r"[\"”],? (?:she|he|I|they|you|we|[A-Z][a-z]+) managed\b(?! to)|\bmanaged,? [\"“]",
         scope="all", case=True),
    dict(level="hard", label="breathed as a tag",
         regex=r"[\"”],? (?:she|he|I|they|you|[A-Z][a-z]+) breath(?:ed|es)\b|\bbreath(?:ed|es),? [\"“]", scope="all", case=True),
    dict(level="check", label="breathed", regex=r"\bbreathed\b", scope="narration"),
    # language about language: talk narrated as words on a page
    dict(level="hard", label="talk as text",
         regex=r"\btrail(?:ed|s|ing) off\b|\bmid-sentence\b"
               r"|\b(?:didn't|couldn't|never|without|not) (?:quite )?finish(?:ed|ing)? (?:the|her|his|my|their|that) (?:sentence|thought|word)\b"
               r"|\bthe (?:rest|end) of (?:the|her|his|my|their|that) sentence\b|\b(?:her|his|my|their) sentence\b"
               r"|\bswallow(?:ed|ing|s)? (?:the|whatever|her|his|my|their) words?\b"
               r"|\b(?:the|that|those|his|her|my|their) (?:words?|syllables?|sentence|phrase) (?:came out|came|died|formed|spilled|tumbled|caught|stuck)\b"
               r"|\b(?:couldn't|could not|unable to|failed to) (?:form|make|string together) (?:words|a sentence|sentences|a word)\b",
         scope="narration"),
    # words given a flight path: they land, hang or sink
    dict(level="hard", label="words that land",
         regex=r"\b(?:the|that|those|his|her|my|their) (?:words?|syllables?|sentence|phrase|question) (?:landed|lands|hung|hangs|hit|sank|settled|arrived|lingered)\b",
         scope="narration"),
    # the narration saying words gave out instead of showing what replaced them
    dict(level="hard", label="words that fail",
         regex=r"\b(?:beyond|below|past|beneath) (?:language|words|syllables|speech)\b"
               r"|\b(?:language|words?|speech) (?:failed|fails|fail|dissolved|dissolves|dissolving|broke apart|fell apart|gave out|deserted|abandoned|left)\b"
               r"|\bwords? (?:couldn't|could not|can't|cannot) (?:capture|describe|express|explain|say|do)\b|\bwords (?:weren't|aren't|were not) enough\b"
               r"|\bno words (?:for|left)\b|\btoo \w+ for words\b|\bwithout (?:language|words)\b"
               r"|\b(?:the )?(?:ability|capacity) to (?:form|make|speak|use) (?:words|sentences|language)\b|\bforgot how to (?:speak|talk|form words)\b",
         scope="narration"),

    # ---------------------------------------------------------------- stock gestures
    dict(level="hard", label="head tilt",
         regex=rf"\b(?:tilt\w*|cock(?:ed|ing|s)?) {PRON} head\b|\bhead tilt\w*",
         scope="narration"),
    # the head tilt in other words; a check, because a bow can be literal
    dict(level="check", label="head tilt, other words",
         regex=rf"\b(?:inclin\w*|dipp?(?:ed|ing|s)?|tipp?(?:ed|ing|s)?|angl(?:ed|ing|es)) {PRON} head\b",
         scope="narration"),
    dict(level="check", label="the look of someone who",
         regex=r"\bthe (?:\w+ )?\w+ of (?:someone|somebody|a (?:man|woman|person|girl|boy|guy|kid|child|soldier|stranger)) who\b",
         scope="narration"),
    dict(level="hard", label="shifted", regex=r"\bshift(?:ed|ing|s)?\b", scope="narration"),
    dict(level="hard", label="hand cliche",
         regex=rf"\bran a hand through {PRON} hair\b|\bpressed a hand to {PRON} mouth\b|\bpressed {PRON} fingertips to {PRON} lips\b",
         scope="narration"),
    dict(level="check", label="face cliche",
         regex=r"\b(?:wicked|devilish|wolfish|predatory|lazy|knowing|feral) (?:grin|smile|smirk)\b|\bsmirk\w*|\bcorner of (?:her|his|my) (?:mouth|lips?) (?:twitch|quirk|curl|lift)\w*|\bjaw (?:set|clench\w*|tighten\w*|lock\w*|tick\w*)|\beyes (?:glint\w*|sparkl\w*|twinkl\w*|flash\w*|darken\w*|narrow\w*|danc\w*)",
         scope="narration"),
    dict(level="check", label="face cliche",
         regex=r"\bsmile (?:still )?play(?:ing|ed|s)? (?:at|on|across|around|about) (?:her|his|my|their|your) (?:lips|mouth)\b"
               r"|\bpupils? (?:were |was |had )?(?:blown|gone|dilated|wide|huge|black)\b|\bblown(?:-out)? pupils\b",
         scope="narration"),
    dict(level="check", label="glisten", regex=r"\bglisten\w*", scope="narration"),
    dict(level="check", label="gesture cliche",
         regex=r"\bpinch\w* the bridge of \w+ nose\b|\breally looked at\b|\bexhal\w* slowly\b|\blet out a (?:slow|long|shaky|deep|soft) breath\b|\bshook (?:her|his|my) head\b|\brolled (?:her|his|my) eyes\b|\bbit (?:her|his|my) (?:lower )?lip\b"
               r"|\bpoint(?:ed|ing) (?:\w+ ){0,3}(?:at|toward|towards|to|around|across)\b",
         scope="narration"),
    dict(level="check", label="sound torn out",
         regex=r"\b(?:ripped|torn|tore|wrenched|punched) (?:out )?(?:of|from) (?:her|his|my|your|their|our) (?:chest|throat|lungs|mouth)\b",
         scope="narration"),
    dict(level="check", label="silence cliche",
         regex=r"\b(?:tense|heavy|awkward|stunned|thick|uncomfortable|comfortable|charged) silence\b|\bsat in silence\b",
         scope="narration"),
    # a taste or smell set under another one ("under the toast, a hint of lemon")
    dict(level="check", label="layered senses",
         regex=r"\b(?:under(?:neath)?|beneath)\b[^.!?\n]{0,60}\b(?:tast\w*|smell\w*|salt\w*|sweet\w*|bitter\w*|sour\w*|musk\w*|gamey|flavou?r\w*)\b"
               r"|\b(?:tast\w*|smell\w*|scent\w*)\b[^.!?\n]{0,40}\b(?:under(?:neath)?|beneath)\b"
               r"|\bat first\b[^.!?\n]{0,15}\bthen\b[^.!?\n]{0,40}\b(?:tast\w*|smell\w*|salt\w*|sweet\w*|bitter\w*|sour\w*|musk\w*|gamey)\b",
         scope="narration"),
    dict(level="hard", label="autopilot", regex=r"\bautopilot\b", scope="narration"),
    dict(level="check", label="machine words for people",
         regex=r"\bmechanical\w*|\bprecision\b|\befficien\w*|\bcalibrat\w*|\bclinical\w*",
         scope="narration"),
    # computer and office words for thinking, noticing and remembering
    dict(level="check", label="office simile",
         regex=r"\b(?:buffer(?:s|ing|ed)?|bandwidth|boot(?:s|ed|ing)? up|loading screen|reboot\w*|recalibrat\w*"
               r"|clos(?:e|ed|es|ing) the tab|(?:filed|files|filing) (?:\w+ )?(?:away|under)|filed (?:it|that|this|them) as|filed as"
               r"|the archive|archived|data points?|same output|error message|fully operational|on autosave|low battery)\b",
         scope="narration"),
    # the narration handling talk as an object
    dict(level="check", label="said it like",
         regex=r"\b(?:say|says|said|saying) (?:it|that|this|the word|those words|the words|his name|her name|my name|your name) like\b"
               r"|\b(?:say|says|said) the word\b",
         scope="narration"),
    # touch narrated as language: a stroke as a sentence, a body that answers
    dict(level="check", label="language metaphor",
         regex=r"\b(?:conversations?|languages?|vocabular(?:y|ies)|grammar|dialects?|sentences?|punctuation|syllables?|paragraphs?) (?:\w+ ){0,3}bodies\b"
               r"|\bbod(?:y|ies) (?:are |were |was )?(?:having|speaking|talking|spoke|spoken|said|says|answered|answers|asked|asks|wrote|writes|replied|replies)\b"
               r"|\b(?:each|every|a|one) (?:stroke|thrust|touch|kiss|movement|roll|grind|squeeze|tap|look) (?:was |is |like )?(?:a|an) (?:sentence|word|question|answer|reply|confession|prayer|promise|apology|vow|punctuation mark|signature|syllable)\b"
               r"|\bpunctuation mark\b",
         scope="narration"),
    dict(level="check", label="doing the thing",
         regex=r"\b(?:do|does|did|doing) (?:its|his|her|their|the|your|my) (?!same\b)[\w-]+ thing\b"
               r"|\b(?:do|does|did|doing) (?:this|that) thing (?:where|when|with)\b",
         scope="narration"),
    # counting up as a dramatic fragment ("One ring. Two.")
    dict(level="check", label="counting up",
         regex=r"(?:(?<=[.!?] )|^)Two(?: [a-z]+)?\.(?= |$)",
         scope="narration", case=True),
    # a label for how someone works ("the calm he brings to every meeting")
    dict(level="check", label="manner label",
         regex=r"\b(?:brings?|brought|bringing|applies|applied) to (?:everything|anything|every \w+|it all)\b",
         scope="narration"),

    # ---------------------------------------------------------------- hooks and non-events
    # a short opener that announces what the next sentence shows
    dict(level="check", label="announcing opener",
         regex=r"^(?:And |But |So )?(?:There (?:was|were)|It was|This was|Here's the thing|The thing was)\b[^.!?\"“”*]{0,32}[.!?](?= [A-Z])|\bNot that (?:it|this|that) was\b|\bI couldn't help it\.",
         scope="narration"),
    # "the one" as a reinforcing tag (", the one who..."); plain pronoun use ("the next one") passes
    dict(level="hard", label="the one who",
         regex=r", the one (?:who|that|which|she|he|they|I|we|you|[A-Z][a-z]+)\b|\bthe (?!next\b)[a-z-]+ one (?:who|that|which|she|he|they|I|we|you|[A-Z][a-z]+)\b",
         scope="narration", case=True),
    dict(level="check", label="and then there was",
         regex=r"\bAnd then there (?:was|were)\b", scope="narration", case=True),
    # narrating, with drama, something that didn't happen
    dict(level="check", label="non-event",
         regex=r"\b(?:didn't|did not|never) (?:even )?(?:look up|look over|notice|react|flinch|turn around|hear a thing|hear)\b|\b(?:nobody|no one) (?:noticed|looked up|moved|reacted)\b|\bnothing happened\b"
               r"|\b(?:it )?(?:was(?:n't| not) |not )an? (?:question|request|suggestion)\b",
         scope="narration"),
    # a face that gets hedged ("what might have been a frown") or seen through ("never reached his eyes")
    dict(level="check", label="hedged face",
         regex=rf"\b(?:what|something that) (?:might|could|may) have been an? \w+"
               rf"|\b(?:didn't|did not|never) reach(?:ed)? (?:{PRON}|the muscles around {PRON}) eyes\b",
         scope="narration"),
    dict(level="check", label="room holding its breath",
         regex=r"\bheld its breath\b|\b(?:the )?(?:room|corridor|hallway|house|air|world|city|station|night|street|building) (?:seemed to (?:wait|listen|hold)|waited|exhaled|sighed)\b",
         scope="narration"),
    # an emotion named through a stock noun ("a twinge of doubt")
    dict(level="check", label="pang of",
         regex=r"\b(?:needle|stab|pang|twinge|flicker|surge|wave|jolt|spike|rush|knot|prickle|thread|ripple) of (?:apprehension|dread|fear|panic|guilt|shame|relief|unease|anxiety|hope|grief|longing|anger|irritation|satisfaction|pride|embarrassment|jealousy|doubt|regret|sadness|annoyance|affection|tenderness)\b",
         scope="narration"),

    # ---------------------------------------------------------------- distance and filtering
    dict(level="check", label="filtering",
         regex=r"\b(?:could(?:n't)?|can't) (?:feel|see|hear|smell|taste)\b|\b(?:felt|saw|heard|noticed|watched|realized) (?:that|how|as)\b",
         scope="narration"),
    dict(level="check", label="noticing verb",
         regex=r"\b(?:clocked|registered|catalog(?:ue)?d|processed)\b|\bnoted\b(?! down)"
               r"|\b(?:clock(?:s|ing)|register(?:s|ing)|catalog(?:ue)?s|catalog(?:u)?ing)\b(?= (?:the|that|this|it|her|his|my|their|your|every|each|how|what|him|me|you|them|as)\b)",
         scope="narration"),
    dict(level="hard", label="inner filler",
         regex=rf"\bcouldn't help but\b|\bfound (?:herself|himself|myself|themselves)\b|\bcouldn't shake the feeling\b|\b(?:a|some) part of (?:her|him|me)\b|\bwithout (?:meaning to|realizing)\b|\bbefore {SUBJ} knew it\b|\bdidn't realize {SUBJ} was\b",
         scope="narration"),

    # ---------------------------------------------------------------- hedges
    # "almost" or "nearly" before an action verb is a near miss, a real event, so it passes; before an adjective,
    # a noun, "like" or a face verb it's a hedge
    dict(level="check", label="hedge",
         regex=r"\b(?:kind of|sort of|somewhat|a bit|a little|slightly|seem(?:ed|s|ing)?(?: to)?|rather|fairly|quite|practically)\b"
               r"|\b(?:almost|nearly)\b(?! (?!(?:smil|laugh|grin|frown|smirk|sob|winc|sigh|chuckl|giggl)\w*)(?:\w+(?:ed|ing)|fell|hit|lost|ran|broke|threw|took|bit|spat|got|came|went|ate|drove|forgot|swung|tore|blew|slid|stood|fell)\b)",
         scope="narration"),
    dict(level="hard", label="half-X hedge",
         regex=r"\bhalf-(?:asleep|awake|joking|teasing|hard|erect|lidded|smil\w*|laugh\w*|serious|heartedly|naked|dressed|expecting)\b|\bhalf (?:a )?(?:smile|laugh|grin)\b|\bhalf \w+, half \w+",
         scope="narration"),
    dict(level="check", label="one single action",
         regex=r"\b(?:gave (?:it|him|her|me) |in |with |for )?one (?:rough|hard|quick|brutal|violent|long|slow|sharp|last|final|deep|swift|single)? ?(?:pull|stroke|hit|thrust|shove|push|slam|tug|yank|drag|swipe|lick|squeeze|kiss|look|glance|line|breath|motion|movement|go)\b",
         scope="narration"),
    # a one-word adjective or adverb sentence hung on the action before it ("He nods. Brief.")
    dict(level="check", label="one-word manner",
         regex=rf"(?<=[.!?] )(?:{ADJ}|{ADV}|present|simple|plain|true|exploratory|gentle"
               r"|(?:slow|soft|quick|fast|deep|hard|long|short|tight|loud|quiet|warm|high|low|light|rough|harsh|smooth|sweet|firm|strong)er"
               r"|closer|gentler|wetter|heavier|louder)\.(?= |$)",
         scope="narration"),
    # flat adverbs ("walked quick"); linking verbs ("went quiet") take an adjective and are skipped
    dict(level="check", label="flat adverb",
         regex=r"\b(?!(?:went|got|fell|turned|stayed|kept|felt|seemed|looked|sounded|grew|ran|need|speed)\b)\w+(?:ed|ing)\s+(?:slow|quick|soft|quiet|loud)\b(?![\w-])"
               r"|\b(?!out\s+loud\b)(?:out|up|down|in|back|off)\s+(?:slow|quick|soft|quiet|loud)\b(?![\w-])",
         scope="narration"),

    # ---------------------------------------------------------------- abstraction and signposting
    dict(level="hard", label="the kind of",
         regex=r"\bthe (?:kind|sort) (?:of|that|who|where)\b", scope="narration"),
    dict(level="check", label="the way",
         regex=r"(?<!\ball )(?<!\bon )(?<!\bby )(?<!\bin )(?<!\bof )(?<!\bout of )(?<!\bhalf)\bthe way\b(?! (?:back|home|down|up|out|over|in|through|across|to (?:the|her|his|my)))",
         scope="narration"),
    dict(level="hard", label="something vague",
         regex=rf"\bsomething (?:shifted|stirred|broke|cracked|snapped|moved|flickered|passed between|unspoken|unreadable|raw|dark|dangerous|primal|deeper|else entirely|she couldn't name|he couldn't name|I couldn't name)\b|\bsomething (?:like|between) (?:grief|hope|anger|relief|fear|longing|pride|guilt)\b|\bsomething (?:in|behind) {PRON} (?:expression|eyes|chest|voice)\b",
         scope="narration"),
    dict(level="check", label="something", regex=r"\bsomething\b", scope="narration"),
    dict(level="check", label="-ness noun",
         regex=r"\b(?:hollowness|emptiness|wrongness|otherness|aliveness|brokenness|rightness|realness)\b",
         scope="narration"),
    # a remark that "lands"; a ball or a shirt landing somewhere is literal and passes
    dict(level="check", label="lands or arrives",
         regex=r"\b(?:words?|line|joke|comment|question|truth|news|compliment) (?:landed|lands|arrived|arrives)\b|\b(?:it|that) (?:landed|lands|arrived|arrives)\b(?! (?:just |right |somewhere |deep |back )?(?:in|on|inside|into|onto|near|beyond|behind|by|at|against|under|over|outside)\b)|\bland(?:ed|s)? (?:hard|differently|wrong|right|softly)\b",
         scope="narration"),
    # "Not X. Not Y. Just Z." and "No X, no Y, just Z"
    dict(level="check", label="not this, not that", stack=True,
         regex=rf"\b{NEG}\b{CL}{{0,60}}?{CL_END}(?:{LEAD}{NEG}\b{CL}{{0,60}}?{CL_END})*{PIVOT}{CL}{{0,60}}"
               rf"|{CL_START}{LEAD}{NEG}\b{CL}{{0,60}}?{CL_END}(?:{LEAD}{NEG}\b{CL}{{0,60}}?{CL_END})+",
         scope="narration"),
    # "not rude, just tired" and "It wasn't luck. It was practice."
    dict(level="check", label="not X but Y", stack=True,
         regex=rf"\b(?:{NEG}|\w+n't)\b[^.!?;\"“”]{{1,50}}?(?:,| -) {PIVOT} [^.!?;,\"“”]{{1,40}}"
               r"|\b(i|you|he|she|it|we|they|this|that) (?:wasn't|isn't|weren't|aren't) [^.!?\"“”]{1,50}\. \1 (?:was|is|were|are)\b"
               r"|\bless [^,.;\"]{1,30} than\b",
         scope="narration"),
    dict(level="check", label="just...", regex=r"\bjust\s?\.\.\.|\.\.\.\s?just\b", scope="all"),

    # ---------------------------------------------------------------- stopwatching
    dict(level="hard", label="stopwatch",
         regex=r"\bfor (?:a|one|one long|a long|a brief|a split) (?:moment|second(?! (?:opinion|helping|look|chance|time|round|try|thought))|beat|heartbeat|breath|instant)\b|\ba (?:second|moment|beat|heartbeat) later\b|\bafter a (?:moment|beat|second|pause|long moment)\b|\bin that (?:moment|instant)\b|\ba beat of silence\b|\bthen,? a beat\b",
         scope="narration"),
    # "a beat" as a unit of time is screenwriter language; a verb object ("beat him at chess") or a music beat passes
    dict(level="hard", label="a beat",
         regex=r"\b(?:a|one|another) beat\b(?! (?:him|her|them|me|you|us|it|that|this|the|my|his|their|our|your|drop|drops|dropped)\b)|\bbeats? (?:late|later|behind|ahead|too)\b",
         scope="narration"),
    # "The second I sat down, the phone..." times the action in units; the ordinal ("the second time") passes
    dict(level="check", label="the second that", case=True,
         regex=r"(?<![Oo]n )\b(?:[Tt]he|[Tt]hat) (?:second|instant|minute) (?=(?:I|you|he|she|it|we|they|my|your|his|her|its|our|their|the|a|an)\b|(?!I\b)[A-Z][a-z]+\b)",
         scope="narration"),

    # ---------------------------------------------------------------- "zero" for "no"
    # an amount nobody counted ("zero interest"); a real zero (degrees, percent, a reading) passes
    dict(level="check", label="zero for no", case=False,
         regex=r"\bzero (?!percent\b|degrees?\b|calories?\b|hour\b|point\b|to\b|out of\b|and\b|or\b|\d)(?=[a-z])",
         scope="narration"),

    # ---------------------------------------------------------------- ownership
    # a body part doing what the person does ("his hand went to the door"); involuntary reactions are left to the
    # reader
    dict(level="check", label="body part as actor", case=True,
         regex=r"(?<!\bin )(?<!\bon )(?<!\bof )\b(?:[Hh]er|[Hh]is|[Mm]y|[Tt]heir|[Yy]our|(?!I\b)[A-Z][a-z]+'s)(?: (?:other|free|left|right|own))? (?:hand|hands|fingers|fingertips|palm|palms|arm|arms|thumb|thumbs|knuckles|fist)(?: (?:still|already|finally|instantly|immediately))? (?:went|came|moved|found|lifted|rose|reached|slid|drifted|crept|wandered|landed|dropped|shot|flew|settled|traveled|crawled|snaked|darted|strayed|made (?:its|their) way)\b",
         scope="narration"),

    # ---------------------------------------------------------------- the cliche catalog
    dict(level="hard", label="breath drama",
         regex=rf"\bbreath {SUBJ} (?:didn't know|hadn't known|hadn't realized)\b|\bforgot how to breathe\b|\bremembered how to breathe\b|\bthe air left {PRON} lungs\b|\bexhaled like a\b",
         scope="narration"),
    dict(level="hard", label="heart drama",
         regex=rf"\bheart (?:did (?:something|a)|lurched|stuttered|climbed|dropped into|cracked open|beat against)\b|\bsomething (?:cracked|tore) (?:open|loose) in {PRON} chest\b|\bache behind {PRON} sternum\b",
         scope="narration"),
    dict(level="check", label="pounding heart",
         regex=r"\bheart (?:hammer|pound|race|skipp|thud|clench|squeez)\w*", scope="narration"),
    dict(level="hard", label="eye cliche",
         regex=rf"\beyes (?:searched|smoldered|burned|went distant|gave nothing away|gave everything away|did something)\b|\bstorm behind {PRON} eyes\b|\blight died in {PRON} eyes\b|\beyes that (?:had seen|held)\b",
         scope="narration"),
    dict(level="hard", label="silence drama",
         regex=r"\bsilence (?:stretched|fell|hung|pressed|settled|wrapped|was deafening)\b|\b(?:pregnant|loaded) pause\b|\bweight of the silence\b",
         scope="narration"),
    dict(level="hard", label="hung in the air", regex=r"\bhung in the air\b|\bair between them\b|\bspace between them\b",
         scope="narration"),
    dict(level="check", label="humorless laugh",
         regex=r"\bhumorless\w*|\bmirthless\w*|\blaugh (?:with no|without) humor\b|\blaugh that wasn't\b",
         scope="narration"),
    dict(level="hard", label="time stops",
         regex=r"\bthe world narrowed\b|\beverything else (?:fell away|stopped mattering)\b|\btime (?:stopped|slowed|stretched|did something|moved strangely|seemed to)\b|\bbetween one heartbeat and the next\b|\bthe edges of things\b",
         scope="narration"),
    dict(level="hard", label="water metaphor",
         regex=r"\b(?:washed|crashed|swept) over (?:her|him|me|them)\b|\bhit (?:her|him|me) like a wave\b|\bflooded (?:her|his|my) (?:senses|body|system|veins)\b|\b(?:tide|undertow) of\b|\bcame up for air\b|\blike a (?:tidal )?wave\b",
         scope="narration"),
    dict(level="hard", label="fire or ice",
         regex=rf"\bheat that had nothing to do with\b|\bfire (?:under|in) {PRON} (?:skin|veins|belly|chest)\b|\bice in {PRON} (?:veins|voice)\b|\bsomething (?:ignited|caught)\b|\bwent up in smoke\b",
         scope="narration"),
    dict(level="hard", label="animal instinct",
         regex=rf"\bthe animal part of\b|\bsomething (?:animal|primal|feral) in\b|\bevery instinct (?:said|screamed)\b|\b{PRON} body knew before\b|\bhair on the back of {PRON} neck\b|\bprimal (?:instincts?|urges?|need)\b|\banimal (?:urges?|instincts?)\b|\blike rabbits\b"
               r"|\blike (?:a|an|some) (?:caged|cornered|trapped|wild|wounded|hunted) (?:animal|beast|thing|creature)\b",
         scope="narration"),
    dict(level="hard", label="long stare",
         regex=r"\bcouldn't look away\b|\bheld the (?:look|gaze) a beat too long\b|\bsomething passed between them\b|\blike (?:she|he) was the only (?:thing|person) in the room\b",
         scope="narration"),
    # Comparisons from courtrooms, tax offices and hostage scenes used for how someone looks or acts. The first
    # rule scans dialogue too; the second keeps everyday words ("it's not a crime") to narration.
    dict(level="check", label="courtroom simile",
         regex=r"\bhostages?\b|\bransom\b|\bproof[- ]of[- ]life\b|\bnegotiators?\b|\blisting (?:her|his|my|your|their) demands\b"
               r"|\b(?:peace|hostage|treaty|ceasefire) (?:talks|treaty|negotiations?|summit)\b|\b(?:international|diplomatic) (?:incident|summit|negotiations?|crisis)\b|\bconven\w+ an? (?:\w+ )?summit\b|\bterms of surrender\b"
               r"|\btax(?:es|man)?\b|\baudit(?:s|ed|ing|ors?)?\b|\baccountants?\b|\bpaperwork\b|\bbureaucra\w*|\bnotari[sz]\w*|\bin triplicate\b|\bfine print\b|\bterms and conditions\b|\binsurance (?:claims?|adjusters?|forms?)\b"
               r"|\b(?:stamp|fill|sign|process)\w* (?:out |in )?(?:a|an|the|some) (?:\w+ )?(?:forms?|waivers?|permission slips?)\b|\bfil(?:e|ed|es|ing) (?:a |an )?(?:formal )?(?:complaint|report|grievance|motion|appeal|claim)\b|\bofficial record\b"
               r"|\bprosecut\w*|\bdefendants?\b|\battorneys?\b|\blawyer\w*|\bcourtroom\b|\bkangaroo court\b|\bcontempt of court\b|\bin court\b|\bcourt[- ](?:case|date|order\w*|reporter|transcript)\b|\btribunals?\b|\bon trial\b|\bmistrial\b|\btrial by \w+|\bin a trial\b"
               r"|\bjur(?:y|ies|ors?)\b|\bverdicts?\b|\byour honou?r\b|\bjurisdiction\w*|\boverruled\b|\bsentenced\b|\bparole\w*|\btestimon\w*|\btestif(?:y|ied|ies|ying)\b|\bunder oath\b|\bdepositions?\b|\baffidavits?\b|\bsubpoena\w*|\bcross-examin\w*"
               r"|\b(?:closing|opening) (?:arguments?|statements?)\b|\bwitness (?:stand|statements?|box|protection)\b|\bsworn (?:statement|testimony)\b|\blegal(?:ly)? (?:argument|brief|counsel|document|statement|defen[cs]e|team|precedent|proceeding|binding|admissible)\w*"
               r"|\bpleaded (?:guilty|not guilty|insanity)\b|\bcase closed\b|\brest(?:s|ed)? (?:her|his|my|their|our|the) case\b|\bbuild\w* (?:a|the|her|his|my) case\b|\b(?-i:Exhibit [A-Z])\b"
               r"|\bcrime[- ]scenes?\b|\bforensic\w*|\bpolice (?:lineup|report|sketch)\b|\b(?:a|the|prime) suspect\b|\bmugshots?\b|\binterrogat\w*|\bchalk outline\b",
         scope="all"),
    dict(level="check", label="courtroom simile",
         regex=r"\bevidence\b|\bcrimes?\b|\bkidnap\w*|\billegal\w*|\bshould(?:n't)? have been legal\b|\bcustody\b|(?<!read )\breceipts?\b|\bloopholes?\b|\btechnicalit(?:y|ies)\b",
         scope="narration"),
    dict(level="hard", label="copper taste",
         regex=r"\btaste of (?:copper|iron)\b|\b(?:copper|iron) on (?:her|his|my) tongue\b|\bmetallic taste\b|\btasted like [\w ]{1,20} and (?:copper|iron|ash|regret|salt|desperation|need|sin|longing|danger)\b",
         scope="narration"),
    # a smell or taste that lists an abstraction beside the real things; real things smell, decisions don't
    dict(level="check", label="abstract smell",
         regex=r"\b(?:(?:smell|scent|stink|reek|taste|stank)(?:s|ed|ing)? (?:like|of)|the (?:smell|scent|taste|stink) of)\b"
               r"[^.!?;\"“”]{0,150}?\b(?:"
               r"sleep|regrets?|decisions?|choices?|promises?|memor(?:y|ies)|home|safety|comfort|belonging|someone|something"
               r"|everything|nothing|possibilit(?:y|ies)|apolog(?:y|ies)|want|need|desire|ambition|grief|hope|fear|trouble|sins?"
               r"|nostalgia|the past|the future|victory|defeat|success|failure|effort|freedom|lives|years|time|mistakes?|danger"
               r"|desperation|longing|intentions?|warmth|summer|childhood|youth|tomorrow|yesterday|forever|beginnings?|endings?"
               r"|goodbyes?|secrets?|lies|truth|love|heartbreak|intimacy|chaos|peace|calm|quiet|silence|ruin|bad news"
               r"|(?!(?:base|pave|apart|compart|gar|oint|instru|condi|sedi)ment|(?:vaca|sta|lo|po|ra|na|mo|loca)tion)[a-z]{4,}(?:ness|tion|ity|ment|ship|hood))\b",
         scope="narration"),
    dict(level="check", label="hum words",
         regex=r"\bsnag\w*|\bhum(?:s|med|ming)?\b|\bdron(?:e|es|ed|ing)\b|\bfrequenc\w*|\bresonan\w*|\bthrumm?\w*",
         scope="narration"),
    dict(level="check", label="vague adjective",
         regex=r"\b(?:bittersweet|ephemeral|ethereal|evocative|gossamer|haunting\w*|incandescent|ineffable|iridescent|kaleidoscopic|liminal|luminous|mercurial|nebulous|otherworldly|poignant\w*|profound\w*|sublime|transcendent|uncanny|visceral\w*|wistful\w*)\b",
         scope="narration"),
    dict(level="check", label="big emotion word",
         regex=r"\b(?:anguish\w*|bereft|desolat\w*|despair\w*|inconsolabl\w*|obliterat\w*|ravaged|torment\w*|wretched\w*)\b",
         scope="narration"),
    dict(level="check", label="shimmer word",
         regex=r"\b(?:shimmer\w*|glimmer\w*|rippl\w*|cascad\w*|unfurl\w*|smolder\w*|radiat\w*)\b",
         scope="narration"),
    dict(level="check", label="intensifier",
         regex=r"\b(?:absolutely|achingly|breathtakingly|devastatingly|heartbreakingly|impossibly|incredibly|indescribably|inexorably|profoundly|unfathomably|unspeakably|utterly)\b",
         scope="narration"),
    dict(level="check", label="whole or entire", regex=r"\b(?:whole|entire)\b", scope="narration"),
    # "mud and all", "all those boxes": is the phrase doing a job, or standing in for the real detail?
    dict(level="check", label="and-all shorthand",
         regex=r"\b\w+ and all\b(?=[,.;:!?)*]|\s*$)"
               r"|\ball (?:that|those|these) (?!(?:is|was|were|had|has|she|he|they|it|i|you|we|mattered|remained|left|happened)\b)[a-z]+"
               r"|\ball (?:two|three|four|five|six|seven|eight|nine|ten|twelve|twenty|fifty|sixty|hundred)\b",
         scope="narration"),
    # words that insist a thing is exact or real ("the exact same bus", "truly useful")
    dict(level="check", label="exactness word",
         regex=r"\b(?:exact|exactly|precise|precisely|genuine|genuinely|truly|entirely|specifically)\b|\bevery single\b",
         scope="narration"),
    dict(level="check", label="genuinely, said", regex=r"\bgenuine(?:ly)?\b", scope="dialogue"),

    # ---------------------------------------------------------------- triples and runs
    # three or more clauses opening on the same word or words, short lists on a common opener, "X and X and X"
    dict(level="check", label="parallel triple", stack=True,
         regex=rf"\b{OPENER}{CL}{{0,80}}?{CL_END}{LEAD}\1\b{CL}{{0,80}}?{CL_END}{LEAD}\1\b{CL}{{0,80}}"
               rf"|\b({DET})(?: [\w'’-]+){{1,2}}{CL_END}{LEAD}\2(?: [\w'’-]+){{1,2}}{CL_END}{LEAD}\2\b{CL}{{0,40}}"
               r"|\b(\w{2,}) and \3 and \3\b",
         scope="narration"),
    # "cold, then wet, then miserable" and "from shy to loud to hoarse"
    dict(level="check", label="then, then", stack=True,
         regex=r"\b[\w'’-]+,? then (?:[\w'’-]+ ){0,2}[\w'’-]+,? then (?:[\w'’-]+ ){0,2}[\w'’-]+|\bfrom [\w'’-]+ to [\w'’-]+ to [\w'’-]+",
         scope="narration"),
    # comma and "and" stacks of adjectives ("a cold, empty room", "quiet and careful")
    dict(level="check", label="adjective stack", stack=True,
         regex=rf"\b{ADJ}(?:, (?:and )?{ADJ}\b)+"
               rf"|(?:,| -) (?:so |too |all )?{ADJ}(?:(?:,| and|, and) {ADJ}\b)*(?:,? and) {ADJ}\b(?:(?:,? and) {ADJ}\b)*"
               rf"|\b{ADV}(?:,? (?:and )?{ADV}\b)+",
         scope="narration"),
    # "X, Y and Z"; "mud and all" is an idiom, so "and all" passes
    dict(level="check", label="rule of three", stack=True,
         regex=r"\b[a-z]+, [a-z]+,? (?:and|or) (?!all\b)[a-z]+\b(?!-)", scope="narration"),
    dict(level="check", label="rule of three", stack=True,
         regex=r"(?:-|,) [a-z]+, [a-z]+, (?:[a-z]+ )?[a-z]+(?=[.!?\"])", scope="narration"),
    # a dash that tacks a list of descriptions onto a finished clause; three or more pieces is a triad
    dict(level="check", label="dash list", stack=True,
         regex=rf"(?:[{DASHES}] ?|(?<=\w) - )(?={DASH_FIRST})(?=[^.!?\"“”{DASHES}]*,[^.!?\"“”{DASHES}]*,)[^.!?\"“”{DASHES},]{{3,}}(?:,(?: and)? {DASH_NOT}[^,.!?\"“”{DASHES}]{{2,}}){{2,}}(?=[.!?{DASHES}]| - |$)",
         scope="narration"),
    dict(level="check", label="dash tail", stack=True,
         regex=rf"(?:[{DASHES}] ?|(?<=\w) - )(?={DASH_FIRST})[^.!?\"“”{DASHES},]{{3,}}(?:,(?: and)? {DASH_NOT}[^,.!?\"“”{DASHES}]{{2,}})(?=[.!?{DASHES}]| - |$)",
         scope="narration"),
    # blocking written as a list of events: three sentences opening on the same subject, with at most one other
    # sentence between them
    dict(level="check", label="same-subject run",
         regex=r"(?:^|(?<=[.!?] ))(She|He|I|They|We)\b[^.!?\"]{3,240}[.!?] (?:[^.!?\"]{3,240}[.!?] )?\1\b[^.!?\"]{3,240}[.!?] (?:[^.!?\"]{3,240}[.!?] )?\1\b",
         scope="narration", case=True),
    dict(level="check", label="three short sentences", stack=True,
         regex=r"(?:^|(?<=[.!?] ))[A-Z][a-z]*(?: [a-z]+){0,2}\. [A-Z][a-z]*(?: [a-z]+){0,2}\. [A-Z][a-z]*(?: [a-z]+){0,2}\.",
         scope="narration", case=True),
    # the dropped subject: a sentence that opens on its verb ("Checks the clock.", "Waiting."). There is no
    # tagger, so it's a verb-shaped first word plus what follows, with a stop list for nouns and adverbs
    dict(level="check", label="dropped subject",
         regex=r"(?:^|(?<=[.!?] )|(?<=[.!?]\* ))(?!(?:" + SUBJLESS_NOT + r")\b)(?!Turns out\b)"
               r"(?:[A-Z][a-z]{2,}(?:s|ed)\b(?= (?:the|a|an|her|his|my|your|their|its|our|it|him|them|me|you|us|at|to|into|onto"
               r"|up|down|back|out|off|over|through|across|against|away|forward|around|once|again|hard|slowly|toward|towards"
               r"|in|on|with|from|for|she|he|i|they|we|that|this|herself|himself|myself|yourself)\b)"
               r"|[A-Z][a-z]{2,}s\.(?= |$)"
               r"|[A-Z][a-z]{2,}ing(?: [a-z'’]+){0,3}\.(?= |$))",
         scope="narration", case=True),
    # chest metaphors ("his chest went tight"); a literal chest is left to the reader
    dict(level="check", label="chest feeling",
         regex=rf"\bwhole chest\b|\b(?:in|inside|behind|through|from) {PRON} chest\b|\b{PRON} chest (?:went|felt|did|was full)\b"
               r"|\bchest (?:tightened|tightening|tight|ached|aching|swelled|swelling|warmed|clenched|squeezed|constricted|caved|loosened|lurched|hollow(?:ed)?)\b",
         scope="narration"),
    # gut reactions reported instead of thought ("his stomach sank")
    dict(level="check", label="gut drop",
         regex=rf"\b(?:{PRON} )?(?:stomach|gut)s? (?:dropped|sank|sunk|plummeted|lurched|flipped|twisted|knotted|clenched|churned|turned over|did a)\b"
               r"|\b(?:pit|knot) (?:of|in) (?:my|his|her|their) (?:stomach|gut)\b|\bgut[- ]punch(?:ed)?\b",
         scope="narration"),
]


# ---------------------------------------------------------------- the stock phrases (phrases.txt)
PHRASES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phrases.txt")


def read_phrases(path=PHRASES_FILE):
    """(phrases, {phrase lowercased: tag}, {tag: family}) from a phrases file in the format its header gives."""
    phrases, tag_of, family_of = [], {}, {}
    tag = None
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return phrases, tag_of, family_of
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("@"):
            name, _, fam = line[1:].partition("|")
            tag = name.strip()
            family_of[tag] = fam.strip() or "cliches"
            continue
        if line.lower() not in tag_of:
            phrases.append(line)
            tag_of[line.lower()] = tag or "banned phrase"
    return phrases, tag_of, family_of


def _phrase_rule(path=PHRASES_FILE):
    phrases, tag_of, family_of = read_phrases(path)
    if not phrases:
        return [], {}
    alts = "|".join(re.escape(p) for p in sorted(phrases, key=len, reverse=True))
    return [dict(level="check", label="banned phrase", regex=rf"\b(?:{alts})\b", scope="narration",
                 ai_relabel=tag_of)], family_of


_PHRASE_RULES, PHRASE_FAMILIES = _phrase_rule()
RULES += _PHRASE_RULES


# ---------------------------------------------------------------- families
FAMILIES = {
    "house rules": ["em dash", "banned word", "banned word (literal?)", "4/7 number", "made-up count"],
    "distance": [
        "voice adjective tag", "voice described", "managed as a tag", "named emotion", "breathed as a tag",
        "breathed", "talk as text", "words that fail", "said it like", "language metaphor", "filtering",
        "noticing verb", "pang of", "body part as actor"],
    "abstraction": [
        "words that land", "lands or arrives", "hung in the air", "silence drama", "-ness noun",
        "hum words", "the kind of", "room holding its breath", "abstract smell"],
    "announcing": ["something vague", "something", "the way", "inner filler", "stopwatch", "a beat",
                    "the second that", "counting up", "sums up its own point"],
    "hedges": ["hedge", "half-X hedge", "hedged face"],
    "patterns": [
        "not this, not that", "not X but Y", "just...", "parallel triple", "then, then",
        "adjective stack", "rule of three", "three short sentences", "dropped subject", "one-word manner",
        "dash list", "dash tail", "the one who", "announcing opener",
        "and then there was", "non-event", "same-subject run"],
    "cliches": [
        "stock AI phrase", "flat adverb", "doing the thing", "manner label", "layered senses",
        "head tilt", "head tilt, other words", "the look of someone who", "shifted", "hand cliche",
        "face cliche", "sound torn out", "glisten", "gesture cliche", "silence cliche", "pounding heart",
        "breath drama", "heart drama", "eye cliche", "humorless laugh", "time stops", "water metaphor",
        "fire or ice", "animal instinct", "long stare", "copper taste", "chest feeling",
        "gut drop", "banned phrase"],
    "wrong register": [
        "machine words for people", "autopilot", "office simile", "courtroom simile",
        "vague adjective", "big emotion word", "shimmer word", "intensifier"],
    "emphasis": ["exactness word", "zero for no", "whole or entire",
                           "and-all shorthand", "genuinely, said", "one single action"],
}
# Every check tag is a call the regex can't make alone; a reader (a person, or the engine's judge) decides it.
JUDGED_LEVEL = "check"

# Rules make_tics leaves out. The scanner this was taken from had house rules that said nothing about AI (one
# project's spelling, names and story conventions); they stayed behind, so nothing is left out here.
HOUSE_ONLY = frozenset()

FAMILY_OF = {label: fam for fam, labels in FAMILIES.items() for label in labels}
for _tag, _fam in PHRASE_FAMILIES.items():
    FAMILY_OF.setdefault(_tag, _fam)
BANNED_LABELS = {"banned phrase"} | set(PHRASE_FAMILIES)
_missing = sorted({r["label"] for r in RULES} - FAMILY_OF.keys())
if _missing:
    raise ValueError(f"tic_patterns: give these labels a family in FAMILIES: {_missing}")
for _r in RULES:
    _r["family"] = FAMILY_OF[_r["label"]]
