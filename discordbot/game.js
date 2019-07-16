card_counts = {
    "m1" : 10,
    "m2" : 5,
    "m3" : 5,
    "s"  : 10,
    "t"  : 20,
    "p"  : 10,
    "n1" : 10,
    "n2" : 8,
    "n3" : 5,
    "w"  : 8,
    "c"  : 5
}

nigiri_scores = {
    "n1" : 1,
    "n2" : 2,
    "n3" : 3
}

maki_counts = {
    "m1" : 1,
    "m2" : 2,
    "m3" : 3
}

start_deck = null


class Game {
    constructor(users) {
        this.users = users;
        this.numplayers = this.users.length;
        this.shz = 12 - this.numplayers;
        this.round = 0;
        if (start_deck === null) {
            
        }
    }

    startRound() {
        this.card = 0;
    }

    onMessage(msg) {

    }
}