package me.iaksh.entity;

import java.sql.Timestamp;

public class Member {
    private Integer memberID;
    private String name;
    private Timestamp membershipStartDate;
    private Timestamp membershipEndDate;

    public String getName() {
        return name;
    }

    public Timestamp getMembershipEndDate() {
        return membershipEndDate;
    }

    public Timestamp getMembershipStartDate() {
        return membershipStartDate;
    }

    public Integer getMemberID() {
        return memberID;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setMemberID(Integer memberID) {
        this.memberID = memberID;
    }

    public void setMembershipEndDate(Timestamp membershipEndDate) {
        this.membershipEndDate = membershipEndDate;
    }

    public void setMembershipStartDate(Timestamp membershipStartDate) {
        this.membershipStartDate = membershipStartDate;
    }
}
