package me.iaksh.test;

public enum Hexagram {
    QIAN(0b111111, "乾为天"),
    KUN(0b000000, "坤为地"),
    ZHUN(0b010001, "水雷屯"),
    MENG(0b100010, "山水蒙"),
    XU(0b010111, "水天需"),
    SONG(0b111010, "天水讼"),
    SHI(0b000010, "地水师"),
    BI(0b010000, "水地比"),
    XIAO_XU(0b110111, "风天小畜"),
    LV(0b111011, "天泽履"),
    TAI(0b000111, "地天泰"),
    PI(0b111000, "天地否"),
    TONG_REN(0b111101, "天火同人"),
    DA_YOU(0b101111, "火天大有"),
    QIAN_(0b000100, "地山谦"),
    YU(0b001000, "雷地豫"),
    SUI(0b011001, "泽雷随"),
    GU(0b100110, "山风蛊"),
    LIN(0b000011, "地泽临"),
    GUAN(0b110000, "风地观"),
    SHI_HE(0b101001, "火雷噬嗑"),
    BI_(0b100101, "山火贲"),
    BO(0b100000, "山地剥"),
    FU(0b000001, "地雷复"),
    WU_WANG(0b111001, "天雷无妄"),
    DA_CHU(0b100111, "山天大畜"),
    YI(0b100001, "山雷颐"),
    DA_GUO(0b011110, "泽风大过"),
    KAN(0b010010, "坎为水"),
    LI(0b101101, "离为火"),
    XIAN(0b011100, "泽山咸"),
    HENG(0b001110, "雷风恒"),
    DUN(0b111100, "天山遁"),
    DA_ZHUANG(0b001111, "雷天大壮"),
    JIN(0b101000, "火地晋"),
    MING_YI(0b000101, "地火明夷"),
    JIA_REN(0b110101, "风火家人"),
    KUI(0b101011, "火泽睽"),
    JIAN(0b010100, "水山蹇"),
    XIE(0b001010, "雷水解"),
    SUN(0b100011, "山泽损"),
    YI_(0b110001, "风雷益"),
    GUAI(0b011111, "泽天夬"),
    GOU(0b111110, "天风姤"),
    CUI(0b011000, "泽地萃"),
    SHENG(0b000110, "地风升"),
    KUN_(0b011010, "泽水困"),
    JING(0b010110, "水风井"),
    GE(0b011101, "泽火革"),
    DING(0b101110, "火风鼎"),
    ZHEN(0b001001, "震为雷"),
    GEN(0b100100, "艮为山"),
    JIAN_(0b110100, "风山渐"),
    GUI_MEI(0b001011, "雷泽归妹"),
    FENG(0b001101, "雷火丰"),
    LV_(0b101100, "火山旅"),
    XUN(0b110110, "巽为风"),
    DUI(0b011011, "兑为泽"),
    HUAN(0b110010, "风水涣"),
    JIE(0b010011, "水泽节"),
    ZHONG_FU(0b110011, "风泽中孚"),
    XIAO_GUO(0b001100, "雷山小过"),
    JI_JI(0b010101, "水火既济"),
    WEI_JI(0b101010, "火水未济");

    // 卦的编号
    private final int code;
    // 卦的名称
    private final String name;

    // 构造方法
    private Hexagram(int code, String name) {
        this.code = code;
        this.name = name;
    }

    // 获取卦的编号
    public int getCode() {
        return code;
    }

    // 获取卦的名称
    public String getName() {
        return name;
    }

    public static Hexagram fromCode(int code) {
        for (Hexagram hexagram : Hexagram.values()) {
            if (hexagram.code == code) {
                return hexagram;
            }
        }
        throw new IllegalArgumentException("Invalid code: " + code);
    }
}