package me.iaksh.controller;

import com.alibaba.fastjson2.JSON;
import me.iaksh.entity.Product;
import me.iaksh.service.ProductService;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/product")
public class ProductController {
    private static ProductService service;

    public static void setService(ProductService service) {
        ProductController.service = service;
    }

    @PostMapping(value = "/update",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String updateProduct(@RequestBody String strProduct) {
        Product product = JSON.parseObject(strProduct,Product.class);
        service.update(product);
        return JSON.toJSONString(product);
    }

    @PostMapping(value = "/insert",
            consumes = "application/json;charset=UTF-8",
            produces = "application/json;charset=UTF-8")
    @ResponseBody
    public String insertProduct(@RequestBody String strProduct) {
        Product product = JSON.parseObject(strProduct,Product.class);
        service.insert(product);
        return JSON.toJSONString(product);
    }

    @GetMapping(value = "/{id}",
            produces = "application/json;charset=UTF-8")
    public String getProductById(@PathVariable Long id) {
        return JSON.toJSONString(service.getById(id));
    }

    @GetMapping(value = "/all",
            produces = "application/json;charset=UTF-8")
    public String getAllProducts() {
        return JSON.toJSONString(service.getAll());
    }

    @GetMapping(value = "/delete/{id}",
            produces = "application/json;charset=UTF-8")
    public String deleteProductById(@PathVariable Long id) {
        service.deleteById(id);
        return "{\n\"status\":\"ok\"\n}";
    }
}
