package me.iaksh.entity;

import java.sql.Date;

public class Member {
    private Integer memberID;
    private String name;
    private Date membershipStartDate;
    private Date membershipEndDate;

    public String getName() {
        return name;
    }

    public Date getMembershipEndDate() {
        return membershipEndDate;
    }

    public Date getMembershipStartDate() {
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

    public void setMembershipEndDate(Date membershipEndDate) {
        this.membershipEndDate = membershipEndDate;
    }

    public void setMembershipStartDate(Date membershipStartDate) {
        this.membershipStartDate = membershipStartDate;
    }
}
