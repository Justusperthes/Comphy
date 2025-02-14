import itertools

def find_combinations(gifts, min_price, max_price):
    """
    Finds all combinations of gifts where the total price is between a minimum and maximum range.
    Gifts cannot occur more than once, and results are sorted by the number of gifts in each combination.
    
    :param gifts: Dictionary of gifts with their prices.
    :param min_price: Minimum allowed total price for the combinations.
    :param max_price: Maximum allowed total price for the combinations.
    :return: List of combinations that meet the condition.
    """
    gift_items = list(gifts.items())
    valid_combinations = []
    
    # Check combinations of all lengths (1 gift to all gifts)
    for r in range(1, len(gift_items) + 1):
        for combo in itertools.combinations(gift_items, r):
            total_price = sum(price for _, price in combo)
            if min_price <= total_price <= max_price:
                valid_combinations.append((combo, total_price))
    
    # Sort the valid combinations by number of gifts, then total price
    valid_combinations.sort(key=lambda x: (len(x[0]), x[1]))
    return valid_combinations

if __name__ == "__main__":
    # Gift data (name: price)
    gifts = {
        "Kai 5\u00bd Brodersaks": 159.95,
        "Lemax Adapter 4.5 V": 165.00,
        "Dobbelt Vaffeljern": 299.00,
        "Tree Delivery": 165.00,
        "Under The Mistletoe Set Of 2": 69.00,
        "Mill Pond": 245.00,
        "Sled With Presents": 39.00,
        "Sharp-Dressed Snowman": 89.00,
        "Santa Feeds Reindeer": 75.00,
        "Mr. And Mrs. Moose": 59.00,
        "Johnnie's Hot Chocolate": 179.00,
        "Winter Adirondack": 39.00,
        "Christmas Eve Visit": 45.00,
    
        "Crepepapir - Bl\u00e5 blanding": 89.00,
        "Kirseb\u00e6rblomster": 129.95
    }
    
    # Minimum and maximum allowed total price
    min_price = 430.00
    max_price = 470.00
    
    # Find all valid combinations
    valid_combos = find_combinations(gifts, min_price, max_price)
    
    # Print the results
    print(f"\nAll combinations of gifts that sum between {min_price} kr and {max_price} kr, sorted by number of gifts and total price:\n")
    for combo, total in valid_combos:
        items = ', '.join(f"{name} ({price} kr)" for name, price in combo)
        print(f"[{items}] => Total: {total:.2f} kr")
    
    print(f"\nTotal combinations found: {len(valid_combos)}")
