package me.iaksh.entity;

import java.math.BigDecimal;
import java.sql.Date;

public class Sales {
    private Integer salesID;
    private Integer productID;
    private Date saleTime;
    private BigDecimal actualUnitPrice;
    private Integer soldQuantity;
    private Integer memberID;

    public void setProductID(Integer productID) {
        this.productID = productID;
    }

    public void setMemberID(Integer memberID) {
        this.memberID = memberID;
    }

    public void setActualUnitPrice(BigDecimal actualUnitPrice) {
        this.actualUnitPrice = actualUnitPrice;
    }

    public void setSalesID(Integer salesID) {
        this.salesID = salesID;
    }

    public void setSaleTime(Date saleTime) {
        this.saleTime = saleTime;
    }

    public void setSoldQuantity(Integer soldQuantity) {
        this.soldQuantity = soldQuantity;
    }

    public Integer getProductID() {
        return productID;
    }

    public Integer getMemberID() {
        return memberID;
    }

    public BigDecimal getActualUnitPrice() {
        return actualUnitPrice;
    }

    public Date getSaleTime() {
        return saleTime;
    }

    public Integer getSalesID() {
        return salesID;
    }

    public Integer getSoldQuantity() {
        return soldQuantity;
    }
}
