package me.iaksh.entity;

public class Source {
    private Integer sourceID;
    private String name;

    public String getName() {
        return name;
    }

    public Integer getSourceID() {
        return sourceID;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setSourceID(Integer sourceID) {
        this.sourceID = sourceID;
    }
}
