package com.trader.trading_backend;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;

@SpringBootApplication
@EnableScheduling  // Enable scheduled tasks
public class TradingBackendApplication {

	public static void main(String[] args) {
		SpringApplication.run(TradingBackendApplication.class, args);
	}

}
