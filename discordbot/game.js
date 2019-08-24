const Discord = require("discord.js");

var card_counts = {
    "m1" : 6,
    "m2" : 12,
    "m3" : 8,
    "s"  : 14,
    "t"  : 14,
    "p"  : 10,
    "n1" : 5,
    "n2" : 10,
    "n3" : 5,
    "w"  : 6,
    // "c"  : 4,
    "d"  : 14
};

var dumping_scores = [
    0,
    1,
    3,
    6,
    10,
    15
];

var nigiri_scores = {
    "n1" : 1,
    "n2" : 2,
    "n3" : 3
};

var maki_counts = {
    "m1" : 1,
    "m2" : 2,
    "m3" : 3
};

function mrand_maxinclusive(min, max) {
    return Math.floor(Math.random() * (max - min + 1) ) + min;
}

function rmovbyval(a, val) {
    for (var x in a) {
        if (a[x] === val) {
            a = a.splice(x, 1);
            return;
        }
    }
}

function dist_score_to_val(scores, nums, key, score) {
    var n = 0;
    for (var x in nums) {
        if (nums[x] === key)
            n++;
    }
    for (var x in nums) {
        if (nums[x] === key) {
            scores[x] += Math.floor(score / n);
            nums[x] = 0;
        }
    }
    return n;
}

class Game {
    constructor(users, channel) {
        this.channel = channel;
        this.users = users;
        this.numplayers = this.users.length;
        this.shz = 12 - this.numplayers;
        this.round = 0;
        this.deck = [];
        this.scores = [];
        this.selections = [];
        this.hands = [];
        this.puddings = [];
        this.state = "starting";
        for (var key in card_counts) {
            for (var i = 0; i < card_counts[key]; i++) {
                this.deck.push(key);
            }
        }
        for (var i = this.deck.length - 1; i > 0; i--) {
            var j = mrand_maxinclusive(0, i);
            var temp = this.deck[j];
            this.deck[j] = this.deck[i];
            this.deck[i] = temp;        
        }

        for (i = 0; i < this.numplayers; i++) {
            this.hands.push([]);
            this.scores.push(0);
            this.selections.push([]);
            this.puddings.push(0);
        }
    }

    getPicks() {
        this.state = "pickwait";
        this.picks = [];
        this.picksRecieved = 0;
        for (var x in this.users) {
            this.users[x].send(`You see ${this.hands[x]}\nReply with the card you want to pick.`);
            this.picks.push(null);
        }
    }

    startRound() {
        this.card = 0;
        this.channel.send(`Round ${this.round + 1} starting!`);
        
        var offset = this.round * this.numplayers;

        for (var i = 0; i < this.numplayers; i++) {
            this.hands[i] = [];
            this.selections[i].length = 0;
            for (var j = (offset + i) * this.shz; j < (offset + i + 1) * this.shz; j++) {
                this.hands[i].push(this.deck[j]);
            }
        }
        
        this.getPicks();
    }

    completePicksReceived() {
        this.state = "calculating";
        var msg = "All picks received! Picks:\n";
        for (var x in this.picks) {
            msg += `${this.users[x].username}: pick ${this.picks[x]}, selection ${this.selections[x]}\n`;
        }
        this.channel.send(msg);

        if (this.round === 1) {
            var startHand = this.hands[this.numplayers - 1];
            for (var i = this.numplayers - 1; i > 0; i--) {
                this.hands[i] = this.hands[i - 1];
            }
            this.hands[0] = startHand;
        } else {
            var startHand = this.hands[0];
            for (var i = 0; i < this.hands.length - 1; i++) {
                this.hands[i] = this.hands[i + 1];
            }
            this.hands[this.numplayers - 1] = startHand;
        }
        this.card++; 
        if (this.card < this.shz) {
            this.getPicks();
        } else {
            this.endRound();
        }
    }

    endRound() {
        this.state = "roundend";
        this.round++;

        var makis = [];
        for (var x in this.users) {
            var wasabi = 0;
            var sashimi = 0;
            var tempura = 0;
            var dumplings = 0;
            makis.push(0);
            for (var c in this.selections[x]) {
                var card = this.selections[x][c];
                if (card === 'w') {
                    wasabi++;
                } else if (card in nigiri_scores) {
                    if (wasabi > 0) {
                        this.scores[x] += nigiri_scores[card] * 3;
                        wasabi--;
                    } 
                    else
                        this.scores[x] += nigiri_scores[card];
                }
                else if (card === 's') {
                    sashimi++;
                    if (sashimi === 3) { 
                        sashimi = 0;
                        this.scores[x] += 10;
                    }
                }
                else if (card === 't') {
                    tempura++;
                    if (tempura === 2) {
                        tempura = 0;
                        this.scores[x] += 5;
                    }
                }
                else if (card === 'd') {
                    dumplings++;
                } else if (card in maki_counts) {
                    makis[x] += maki_counts[card];
                } else if (card === "p") {
                    this.puddings[x]++;
                }
            }
            if (dumplings > 5)
                dumplings = 5;
            this.scores[x] += dumping_scores[dumplings];
        }

        var gameover = this.round === 3;
        
        var n = dist_score_to_val(this.scores, makis, Math.max(makis), 6);
        if (n === 1) {
            dist_score_to_val(this.scores, makis, Math.max(makis), 3);
        }
        if (gameover) {
            if (this.numplayers > 2) {
                dist_score_to_val(this.scores, this.puddings, Math.min(this.puddings), -6);
            }
            dist_score_to_val(this.scores, this.puddings, Math.max(this.puddings), 6);
        }
        var ground = gameover ? "Game" : "Round";
        var msg = `${ground} ended! Standings : \n`; 
        for (var x in this.users) {
            msg += `${this.users[x].username}: points ${this.scores[x]}, puddings ${this.puddings[x]}\n`;
        }
        this.channel.send(msg);
        if (!gameover) {
            this.startRound(); 
        } else {
            this.channel.send("Game over!");
            this.channel.send("stop game");
        }
    }

    onMessage(msg) {
        switch (this.state) {
            case "pickwait":
                for (var x in this.users) {
                    if (msg.channel instanceof Discord.DMChannel && msg.author === this.users[x]) {
                        if (this.picks[x] !== null) {
                            msg.reply("Already got your pick! Wait until the next round.");
                            return;
                        }
                        if (this.hands[x].includes(msg.content)) {
                            this.selections[x].push(msg.content);
                            this.picks[x] = msg.content;
                            rmovbyval(this.hands[x], msg.content);
                            this.picksRecieved++;
                            msg.reply("Got it! Please wait until the next round!");
                            if (this.picksRecieved >= this.numplayers) {
                                this.completePicksReceived();
                            }
                        } else {
                            msg.reply("Please respond with a card in the list I just gave you!");
                        }
                    }
                }
                break;
        }
    }
}

module.exports = Game;