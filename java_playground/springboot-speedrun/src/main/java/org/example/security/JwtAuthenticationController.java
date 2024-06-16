package org.example.controller;

import org.example.security.JwtService;
import org.example.entity.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UserDetailsService;
import org.springframework.web.bind.annotation.*;

@RestController
public class JwtAuthenticationController {

    private final AuthenticationManager authenticationManager;
    private final UserDetailsService userDetailsService;
    private final JwtService jwtService;

    @Autowired
    public JwtAuthenticationController(AuthenticationManager authenticationManager, UserDetailsService userDetailsService, JwtService jwtService) {
        this.authenticationManager = authenticationManager;
        this.userDetailsService = userDetailsService;
        this.jwtService = jwtService;
    }

    @PostMapping("/auth")
    public ResponseEntity<?> createAuthenticationToken(@RequestBody User authenticationRequest) throws Exception {
        try {
            authenticationManager.authenticate(
                    // 两个参数分别是用户名和密码
                    new UsernamePasswordAuthenticationToken(authenticationRequest.getName(),authenticationRequest.getEmail())
            );
        } catch (Exception e) {
            throw new Exception("Incorrect username or password", e);
        }

        final UserDetails userDetails = userDetailsService
                .loadUserByUsername(authenticationRequest.getName());

        final String jwt = jwtService.generateToken(userDetails);

        return ResponseEntity.ok(jwt);
    }
}
