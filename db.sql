drop database if exists fish;
create database fish;
use fish;

create table users (
    id INT PRIMARY KEY AUTO_INCREMENT, 
    email VARCHAR(50), 
    password VARCHAR(50)
    );
