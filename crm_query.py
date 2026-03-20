import sqlite3

DB_PATH = "data/crm.db"


def buscar_cliente(nombre):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, name, phone, website, city
    FROM accounts
    WHERE name LIKE ?
    """, ("%" + nombre + "%",))

    result = cursor.fetchone()

    conn.close()

    return result


def obtener_notas(cliente_id):

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT content, created_time
    FROM notes
    WHERE parent_id LIKE ?
    """, ("%" + cliente_id + "%",))

    notas = cursor.fetchall()

    conn.close()

    return notas


def analizar_cliente(nombre):

    cliente = buscar_cliente(nombre)

    if not cliente:
        print("Cliente no encontrado")
        return

    cliente_id, name, phone, website, city = cliente

    print("\nCLIENTE")
    print(name)
    print("Tel:", phone)
    print("Web:", website)
    print("Ciudad:", city)

    notas = obtener_notas(cliente_id)

    print("\nNOTAS:")

    if not notas:
        print("No hay notas registradas")
        return

    for n in notas:

        content, fecha = n

        print("\nFecha:", fecha)
        print(content)


if __name__ == "__main__":

    cliente = input("Nombre del cliente: ")

    analizar_cliente(cliente)