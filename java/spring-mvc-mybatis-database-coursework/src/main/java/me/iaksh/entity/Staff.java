package me.iaksh.entity;

import java.math.BigDecimal;

public class Staff {
    private Integer staffID;
    private String name;
    private String gender;
    private Integer age;
    private BigDecimal monthlySalary;

    public BigDecimal getMonthlySalary() {
        return monthlySalary;
    }

    public Integer getStaffID() {
        return staffID;
    }

    public Integer getAge() {
        return age;
    }

    public String getGender() {
        return gender;
    }

    public String getName() {
        return name;
    }

    public void setAge(Integer age) {
        this.age = age;
    }

    public void setGender(String gender) {
        this.gender = gender;
    }

    public void setMonthlySalary(BigDecimal monthlySalary) {
        this.monthlySalary = monthlySalary;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setStaffID(Integer staffID) {
        this.staffID = staffID;
    }
}
