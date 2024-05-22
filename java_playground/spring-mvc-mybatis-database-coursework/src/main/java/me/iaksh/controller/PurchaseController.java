package me.iaksh.controller;

import com.alibaba.fastjson2.JSON;
import me.iaksh.entity.Purchase;
import me.iaksh.service.PurchaseService;
import org.springframework.web.bind.annotation.*;

import java.sql.Timestamp;

@RestController
@RequestMapping("/purchase")
public class PurchaseController {
    private static PurchaseService service;

    public static void setService(PurchaseService service) {
        PurchaseController.service = service;
    }

    @PostMapping(value = "/update",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String updatePurchase(@RequestBody String strPurchase) {
        Purchase purchase = JSON.parseObject(strPurchase,Purchase.class);
        service.update(purchase);
        return JSON.toJSONString(purchase);
    }

    @PostMapping(value = "/insert",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String insertPurchase(@RequestBody String strPurchase) {
        Purchase purchase = JSON.parseObject(strPurchase,Purchase.class);
        if (purchase.getPurchaseTime() == null) {
            purchase.setPurchaseTime(new Timestamp(System.currentTimeMillis()));
        }
        service.purchaseProduct(purchase);
        return JSON.toJSONString(purchase);
    }

    @GetMapping(value = "/{id}",
            produces = "application/json;charset=UTF-8")
    public String getPurchaseById(@PathVariable Long id) {
        return JSON.toJSONString(service.getById(id));
    }

    @GetMapping(value = "/all",
            produces = "application/json;charset=UTF-8")
    public String getAllPurchases() {
        return JSON.toJSONString(service.getAll());
    }

    @GetMapping(value = "/delete/{id}",
            produces = "application/json;charset=UTF-8")
    public String deletePurchaseById(@PathVariable Long id) {
        service.deleteById(id);
        return "{\n\"status\":\"ok\"\n}";
    }
}
