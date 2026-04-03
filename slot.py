import random


# Get starting balance

def get_starting_balance():
    while True:
        try:
            balance = int(input("Enter your starting balance: रु."))
            if balance <= 0:
                print("Balance must be a positive number.")
            else:
                return balance
        except ValueError:
            print("Please enter a valid number.")



# Get bet amount
def get_bet_amount(balance):
    while True:
        try:
            bet = int(input("Enter your bet amount: रु."))
            if bet <= 0 or bet > balance:
                print(f"Invalid bet. Enter a value between रु.1 and रु.{balance}.")
            else:
                return bet
        except ValueError:
            print("Please enter a valid number.")


# Spin reels
def spin_reels():
    symbols = ["🍒", "🍋", "🔔", "⭐", "🍉"]
    return [random.choice(symbols) for _ in range(3)]


# Display reels
def display_reels(reels):
    print(f"{reels[0]} | {reels[1]} | {reels[2]}")


# Calculate payout
def calculate_payout(reels, bet):
    # 3 match
    if reels[0] == reels[1] == reels[2]:
        return bet * 10

    # 2 match
    if (reels[0] == reels[1] or
        reels[0] == reels[2] or
        reels[1] == reels[2]):
        return bet * 2

    return 0


# Main game
def main():
    print("🎰 Welcome to the Slot Machine Game!")

    balance = get_starting_balance()
    print(f"You start with a balance of रु.{balance}\n")

    while balance > 0:
        print(f"Current balance: रु.{balance}")

        bet = get_bet_amount(balance)

        reels = spin_reels()
        display_reels(reels)

        payout = calculate_payout(reels, bet)

        if payout > 0:
            print(f"You won रु.{payout}!")
        else:
            print("You lost!")

        # Update balance
        balance += payout - bet

        if balance <= 0:
            print("You are out of money! Game over.")
            break

        # Ask to continue
        play_again = input("Play again? (y/n): ").lower()
        if play_again != "y":
            print(f"You walk away with रु.{balance}.")
            break


# Run program
if __name__ == "__main__":
    main()