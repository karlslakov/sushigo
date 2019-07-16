const Discord = require('discord.js');
const Game = require('./game.js');
const client = new Discord.Client();
const fs = require('fs');

var state = 'idle';
var players = -1;
var users = []
var game = null;

function resetToIdle() {
    state = 'idle';
    players = -1;
    users = []
}

client.on('ready', () => {
 console.log(`Logged in as ${client.user.tag}!`);
 });

client.on('message', msg => {
    switch (state) {
        case 'idle':
            if (msg.content.startsWith('start game')) {
                if (msg.content.endsWith('p2')) {
                    players = 2;
                } else if (msg.content.endsWith('p3')) {
                    players = 3;
                } else if (msg.content.endsWith('p4')) {
                    players = 4;
                } else if (msg.content.endsWith('p5')) {
                    players = 5;
                }
                else {
                    msg.reply("please specify amount of players by saying start game p2/p3/p4/p5");
                    return;
                }
                state = 'rolecall';
                msg.channel.send(`Starting game with ${players} players! Starting rolecall! Type 'here!' (without quotes) to join the game! Type 'cancel game' to cancel!`);
            }
            break;
        case 'rolecall':
            if (msg.content == "cancel game") {
                msg.channel.send("Game canceled!");
                resetToIdle();
            }
            if (msg.content === "here!")
            {
                if (users.includes(msg.author)) {
                    msg.reply("You're already in the game!");
                    return;
                }
                users.push(msg.author);
                var rep = `User ${msg.author.username} has joined the game! \nPlayers : \n`;
                for (user in users) {
                    rep += users[user].username + "\n";
                }
                msg.channel.send(rep);

                if (users.length === players) {
                    msg.channel.send("All players are ready! Starting game!");
                    state = 'game';
                }
            }
            break;
    }
 });

 fs.readFile('token.pword', 'utf8', function(err, contents) {
    client.login(contents)
});
