"""
《天地之中 · 郑州》—— 旁白文案（中英对照）
每个场景是一组字幕行；每行单独合成一段旁白，时间轴由 narrate.py 生成。
"""

TITLE_ZH = "天地之中 · 郑州"
TITLE_EN = "Zhengzhou: Centre of Heaven and Earth"

# (场景 id, [(中文, English), ...])
SCENES = [
    ("river", [
        ("黄河流过邙山脚下，从这里进入下游，奔向平原。",
         "The Yellow River flows past Mount Mang, entering its lower reaches here on its way to the plain."),
        ("这片中原腹地，就是郑州。",
         "This heartland of the Central Plains is Zhengzhou."),
        ("它的故事，要从八千年前说起。",
         "Its story begins eight thousand years ago."),
    ]),
    ("prehistory", [
        ("在新郑裴李岗，先民磨制石器、种下粟米；",
         "At Peiligang, early people ground stone tools and planted millet;"),
        ("大河村彩陶上的太阳纹，记录着五千年前的天空。",
         "sun motifs on Dahecun pottery recorded the sky of five thousand years ago."),
        ("传说中，人文始祖黄帝也诞生在这片土地。",
         "Legend says the Yellow Emperor was born on this land."),
    ]),
    ("shang", [
        ("约三千六百年前，商朝在这里筑起都城。",
         "About 3,600 years ago, the Shang dynasty built a capital here."),
        ("近七公里的夯土城墙，至今仍立在城市中心，",
         "Nearly seven kilometres of rammed-earth walls still stand downtown,"),
        ("出土的青铜方鼎，见证着早期中国的模样。",
         "and bronze cauldrons unearthed here show what early China looked like."),
    ]),
    ("names", [
        ("周代，这里是管国，郑州至今别称“管城”。",
         "Under the Zhou it was the State of Guan, so Zhengzhou is still called Guancheng."),
        ("春秋战国，郑国、韩国先后建都新郑。",
         "In the Spring and Autumn and Warring States eras, Zheng and Han made Xinzheng their capital."),
        ("公元五八三年，隋朝设立郑州，名字沿用至今。",
         "In 583 AD, the Sui dynasty named it Zhengzhou, a name it still bears."),
    ]),
    ("culture", [
        ("这里文脉绵长：诗圣杜甫生于巩义，白居易生于新郑；",
         "Its culture runs deep: the poet Du Fu was born in Gongyi, Bai Juyi in Xinzheng;"),
        ("嵩山脚下的少林寺，晨钟暮鼓已有一千五百多年。",
         "at the foot of Mount Song, Shaolin Temple's bells have rung for over 1,500 years."),
    ]),
    ("railway", [
        ("二十世纪初，京汉、汴洛两条铁路在这里交汇，",
         "In the early 1900s, the Beijing–Hankou and Kaifeng–Luoyang railways met here,"),
        ("一座小县城，被火车拉成了交通枢纽。",
         "and trains turned a small county town into a transport hub."),
        ("一九二三年的二七大罢工，让二七塔成为城市的记忆。",
         "The February 7th strike of 1923 made the Erqi Tower the city's landmark of memory."),
    ]),
    ("modern", [
        ("一九五四年，河南省会从开封迁到郑州，",
         "In 1954, Henan's provincial capital moved from Kaifeng to Zhengzhou,"),
        ("国棉厂的织机声日夜不停，它有了“纺织城”的名号，",
         "its cotton mills hummed day and night, earning it the name \"Textile City\","),
        ("这座城市从此一路生长。",
         "and the city has been growing ever since."),
    ]),
    ("today", [
        ("今天，“米”字形高铁网从这里通向四面八方，",
         "Today, a star-shaped high-speed rail network radiates from here in every direction,"),
        ("航空港连接世界，常住人口超过一千三百万。",
         "its airport links the world, and more than thirteen million people call it home."),
    ]),
    ("ending", [
        ("从裴李岗的炊烟，到高铁的汽笛，",
         "From the cooking smoke of Peiligang to the whistle of bullet trains,"),
        ("郑州始终站在中华文明的十字路口。",
         "Zhengzhou has always stood at the crossroads of Chinese civilisation."),
        ("这，就是天地之中的郑州。",
         "This is Zhengzhou, the centre of heaven and earth."),
    ]),
]
